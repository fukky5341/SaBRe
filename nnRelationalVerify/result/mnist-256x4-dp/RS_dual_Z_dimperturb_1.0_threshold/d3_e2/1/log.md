## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 3.518859945


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398)
1: (-1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865)
2: (-1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576)
3: (-1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979)
4: (-1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516)
5: (-1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349)
6: (-1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487)
7: (-1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090)
8: (-2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340)
9: (-1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.43 + 3.69 = 5.12 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -3.7040631, upper bound: 3.7040631

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5867620, upper bound: 3.5867620
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5867620, upper bound: 3.5867620
time: 1.27 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.68 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.68
Output dim: 8, lower bound: -3.5867620, upper bound: 3.5867620
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.68
Output dim: 8, lower bound: -3.5867620, upper bound: 3.5867620

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5866142, upper bound: 3.5866149
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5866149, upper bound: 3.5866143
time: 1.23 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5866142, upper bound: 3.5866149
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5866149, upper bound: 3.5866143
time: 1.23 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 5.82 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 5.82
Output dim: 8, lower bound: -3.5866142, upper bound: 3.5866149
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 5.82
Output dim: 8, lower bound: -3.5866149, upper bound: 3.5866143
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 5.82
Output dim: 8, lower bound: -3.5866142, upper bound: 3.5866149
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 5.82
Output dim: 8, lower bound: -3.5866149, upper bound: 3.5866143

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5866038, upper bound: 3.5866150
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5866142, upper bound: 3.5866037
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5866037, upper bound: 3.5866143
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5866149, upper bound: 3.5866038
time: 1.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5866038, upper bound: 3.5866150
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5866142, upper bound: 3.5866037
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5866037, upper bound: 3.5866143
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5866149, upper bound: 3.5866038
time: 1.32 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 6.04 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.04
Output dim: 8, lower bound: -3.5866038, upper bound: 3.5866150
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.04
Output dim: 8, lower bound: -3.5866142, upper bound: 3.5866037
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.04
Output dim: 8, lower bound: -3.5866037, upper bound: 3.5866143
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.04
Output dim: 8, lower bound: -3.5866149, upper bound: 3.5866038
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.04
Output dim: 8, lower bound: -3.5866038, upper bound: 3.5866150
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.04
Output dim: 8, lower bound: -3.5866142, upper bound: 3.5866037
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.04
Output dim: 8, lower bound: -3.5866037, upper bound: 3.5866143
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.04
Output dim: 8, lower bound: -3.5866149, upper bound: 3.5866038

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5866038, upper bound: 3.5865796
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5865686, upper bound: 3.5866149
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5866142, upper bound: 3.5865681
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5865793, upper bound: 3.5866037
time: 1.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5866037, upper bound: 3.5865793
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5865681, upper bound: 3.5866143
time: 1.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5866149, upper bound: 3.5865686
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5865796, upper bound: 3.5866038
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5866038, upper bound: 3.5865796
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5865686, upper bound: 3.5866149
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5866142, upper bound: 3.5865681
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5865793, upper bound: 3.5866037
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5866037, upper bound: 3.5865793
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5865681, upper bound: 3.5866143
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5866149, upper bound: 3.5865686
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5865796, upper bound: 3.5866038
time: 1.30 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 5.87 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 8, lower bound: -3.5866038, upper bound: 3.5865796
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 8, lower bound: -3.5865686, upper bound: 3.5866149
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 8, lower bound: -3.5866142, upper bound: 3.5865681
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 8, lower bound: -3.5865793, upper bound: 3.5866037
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 8, lower bound: -3.5866037, upper bound: 3.5865793
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 8, lower bound: -3.5865681, upper bound: 3.5866143
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 8, lower bound: -3.5866149, upper bound: 3.5865686
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 8, lower bound: -3.5865796, upper bound: 3.5866038
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 8, lower bound: -3.5866038, upper bound: 3.5865796
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 8, lower bound: -3.5865686, upper bound: 3.5866149
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 8, lower bound: -3.5866142, upper bound: 3.5865681
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 8, lower bound: -3.5865793, upper bound: 3.5866037
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 8, lower bound: -3.5866037, upper bound: 3.5865793
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 8, lower bound: -3.5865681, upper bound: 3.5866143
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 8, lower bound: -3.5866149, upper bound: 3.5865686
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 8, lower bound: -3.5865796, upper bound: 3.5866038

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5805616, upper bound: 3.5805408
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5805564, upper bound: 3.5805505
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5805400, upper bound: 3.5805655
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5805322, upper bound: 3.5805701
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5805697, upper bound: 3.5805315
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5805648, upper bound: 3.5805400
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5805495, upper bound: 3.5805564
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5805402, upper bound: 3.5805611
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5805611, upper bound: 3.5805401
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5805564, upper bound: 3.5805495
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5805400, upper bound: 3.5805648
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5805315, upper bound: 3.5805697
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5805701, upper bound: 3.5805322
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5805655, upper bound: 3.5805400
time: 1.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5805505, upper bound: 3.5805564
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5805408, upper bound: 3.5805616
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5805616, upper bound: 3.5805408
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5805564, upper bound: 3.5805505
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5805400, upper bound: 3.5805655
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5805322, upper bound: 3.5805701
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5805697, upper bound: 3.5805315
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5805648, upper bound: 3.5805400
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5805495, upper bound: 3.5805564
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5805402, upper bound: 3.5805611
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5805611, upper bound: 3.5805401
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5805564, upper bound: 3.5805495
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5805400, upper bound: 3.5805648
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5805315, upper bound: 3.5805697
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5805701, upper bound: 3.5805322
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5805655, upper bound: 3.5805400
time: 1.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5805505, upper bound: 3.5805564
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5805408, upper bound: 3.5805616
time: 1.13 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 5.71 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.71
Output dim: 8, lower bound: -3.5805616, upper bound: 3.5805408
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.71
Output dim: 8, lower bound: -3.5805564, upper bound: 3.5805505
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.71
Output dim: 8, lower bound: -3.5805400, upper bound: 3.5805655
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.71
Output dim: 8, lower bound: -3.5805322, upper bound: 3.5805701
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.71
Output dim: 8, lower bound: -3.5805697, upper bound: 3.5805315
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.71
Output dim: 8, lower bound: -3.5805648, upper bound: 3.5805400
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.71
Output dim: 8, lower bound: -3.5805495, upper bound: 3.5805564
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.71
Output dim: 8, lower bound: -3.5805402, upper bound: 3.5805611
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.71
Output dim: 8, lower bound: -3.5805611, upper bound: 3.5805401
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.71
Output dim: 8, lower bound: -3.5805564, upper bound: 3.5805495
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.71
Output dim: 8, lower bound: -3.5805400, upper bound: 3.5805648
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.71
Output dim: 8, lower bound: -3.5805315, upper bound: 3.5805697
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.71
Output dim: 8, lower bound: -3.5805701, upper bound: 3.5805322
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.71
Output dim: 8, lower bound: -3.5805655, upper bound: 3.5805400
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.71
Output dim: 8, lower bound: -3.5805505, upper bound: 3.5805564
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.71
Output dim: 8, lower bound: -3.5805408, upper bound: 3.5805616
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.71
Output dim: 8, lower bound: -3.5805616, upper bound: 3.5805408
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.71
Output dim: 8, lower bound: -3.5805564, upper bound: 3.5805505
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.71
Output dim: 8, lower bound: -3.5805400, upper bound: 3.5805655
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.71
Output dim: 8, lower bound: -3.5805322, upper bound: 3.5805701
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.71
Output dim: 8, lower bound: -3.5805697, upper bound: 3.5805315
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.71
Output dim: 8, lower bound: -3.5805648, upper bound: 3.5805400
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.71
Output dim: 8, lower bound: -3.5805495, upper bound: 3.5805564
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.71
Output dim: 8, lower bound: -3.5805402, upper bound: 3.5805611
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.71
Output dim: 8, lower bound: -3.5805611, upper bound: 3.5805401
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.71
Output dim: 8, lower bound: -3.5805564, upper bound: 3.5805495
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.71
Output dim: 8, lower bound: -3.5805400, upper bound: 3.5805648
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.71
Output dim: 8, lower bound: -3.5805315, upper bound: 3.5805697
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.71
Output dim: 8, lower bound: -3.5805701, upper bound: 3.5805322
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.71
Output dim: 8, lower bound: -3.5805655, upper bound: 3.5805400
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.71
Output dim: 8, lower bound: -3.5805505, upper bound: 3.5805564
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.71
Output dim: 8, lower bound: -3.5805408, upper bound: 3.5805616

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552487, upper bound: 3.5551954
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552178, upper bound: 3.5552134
time: 1.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552386, upper bound: 3.5552042
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552131, upper bound: 3.5552282
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552230, upper bound: 3.5552227
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5551913, upper bound: 3.5552420
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552089, upper bound: 3.5552270
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5551856, upper bound: 3.5552527
time: 1.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552529, upper bound: 3.5551831
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552267, upper bound: 3.5552060
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552427, upper bound: 3.5551903
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552227, upper bound: 3.5552221
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552282, upper bound: 3.5552124
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552042, upper bound: 3.5552364
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552144, upper bound: 3.5552172
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5551958, upper bound: 3.5552471
time: 1.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552471, upper bound: 3.5551958
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552172, upper bound: 3.5552144
time: 1.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552364, upper bound: 3.5552042
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552124, upper bound: 3.5552282
time: 1.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552221, upper bound: 3.5552227
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5551903, upper bound: 3.5552427
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552060, upper bound: 3.5552266
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5551831, upper bound: 3.5552528
time: 1.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552527, upper bound: 3.5551855
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552270, upper bound: 3.5552089
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552421, upper bound: 3.5551913
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552227, upper bound: 3.5552230
time: 1.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552282, upper bound: 3.5552131
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552042, upper bound: 3.5552386
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552134, upper bound: 3.5552178
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5551954, upper bound: 3.5552487
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552487, upper bound: 3.5551954
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552178, upper bound: 3.5552134
time: 1.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552386, upper bound: 3.5552041
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552131, upper bound: 3.5552282
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552230, upper bound: 3.5552227
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5551913, upper bound: 3.5552420
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552089, upper bound: 3.5552270
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5551856, upper bound: 3.5552527
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552529, upper bound: 3.5551831
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552267, upper bound: 3.5552060
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552427, upper bound: 3.5551903
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552227, upper bound: 3.5552221
time: 1.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552282, upper bound: 3.5552124
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552042, upper bound: 3.5552364
time: 1.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552144, upper bound: 3.5552172
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5551958, upper bound: 3.5552471
time: 1.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552471, upper bound: 3.5551958
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552172, upper bound: 3.5552144
time: 1.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552364, upper bound: 3.5552042
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552124, upper bound: 3.5552282
time: 1.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552221, upper bound: 3.5552227
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5551903, upper bound: 3.5552427
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552060, upper bound: 3.5552266
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5551831, upper bound: 3.5552528
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552527, upper bound: 3.5551855
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552270, upper bound: 3.5552089
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552421, upper bound: 3.5551913
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552227, upper bound: 3.5552230
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552282, upper bound: 3.5552131
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552042, upper bound: 3.5552386
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5552134, upper bound: 3.5552178
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5551954, upper bound: 3.5552487
time: 1.19 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 6.14 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552487, upper bound: 3.5551954
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552178, upper bound: 3.5552134
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552386, upper bound: 3.5552042
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552131, upper bound: 3.5552282
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552230, upper bound: 3.5552227
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5551913, upper bound: 3.5552420
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552089, upper bound: 3.5552270
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5551856, upper bound: 3.5552527
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552529, upper bound: 3.5551831
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552267, upper bound: 3.5552060
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552427, upper bound: 3.5551903
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552227, upper bound: 3.5552221
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552282, upper bound: 3.5552124
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552042, upper bound: 3.5552364
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552144, upper bound: 3.5552172
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5551958, upper bound: 3.5552471
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552471, upper bound: 3.5551958
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552172, upper bound: 3.5552144
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552364, upper bound: 3.5552042
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552124, upper bound: 3.5552282
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552221, upper bound: 3.5552227
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5551903, upper bound: 3.5552427
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552060, upper bound: 3.5552266
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5551831, upper bound: 3.5552528
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552527, upper bound: 3.5551855
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552270, upper bound: 3.5552089
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552421, upper bound: 3.5551913
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552227, upper bound: 3.5552230
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552282, upper bound: 3.5552131
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552042, upper bound: 3.5552386
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552134, upper bound: 3.5552178
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5551954, upper bound: 3.5552487
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552487, upper bound: 3.5551954
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552178, upper bound: 3.5552134
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552386, upper bound: 3.5552041
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552131, upper bound: 3.5552282
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552230, upper bound: 3.5552227
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5551913, upper bound: 3.5552420
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552089, upper bound: 3.5552270
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5551856, upper bound: 3.5552527
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552529, upper bound: 3.5551831
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552267, upper bound: 3.5552060
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552427, upper bound: 3.5551903
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552227, upper bound: 3.5552221
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552282, upper bound: 3.5552124
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552042, upper bound: 3.5552364
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552144, upper bound: 3.5552172
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5551958, upper bound: 3.5552471
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552471, upper bound: 3.5551958
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552172, upper bound: 3.5552144
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552364, upper bound: 3.5552042
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552124, upper bound: 3.5552282
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552221, upper bound: 3.5552227
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5551903, upper bound: 3.5552427
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552060, upper bound: 3.5552266
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5551831, upper bound: 3.5552528
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552527, upper bound: 3.5551855
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552270, upper bound: 3.5552089
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552421, upper bound: 3.5551913
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552227, upper bound: 3.5552230
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552282, upper bound: 3.5552131
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552042, upper bound: 3.5552386
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5552134, upper bound: 3.5552178
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 8, lower bound: -3.5551954, upper bound: 3.5552487

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967739, upper bound: 3.4967370
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967739, upper bound: 3.4967370
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967627, upper bound: 3.4967406
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967627, upper bound: 3.4967406
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967702, upper bound: 3.4967419
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967702, upper bound: 3.4967419
time: 1.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967591, upper bound: 3.4967475
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967591, upper bound: 3.4967475
time: 1.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967469, upper bound: 3.4967667
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967469, upper bound: 3.4967667
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967403, upper bound: 3.4967731
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967403, upper bound: 3.4967731
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967414, upper bound: 3.4967689
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967414, upper bound: 3.4967689
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967367, upper bound: 3.4967774
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967367, upper bound: 3.4967774
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967774, upper bound: 3.4967355
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967774, upper bound: 3.4967355
time: 1.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967684, upper bound: 3.4967401
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967684, upper bound: 3.4967401
time: 1.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967737, upper bound: 3.4967402
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967737, upper bound: 3.4967402
time: 1.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967664, upper bound: 3.4967469
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967664, upper bound: 3.4967469
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967477, upper bound: 3.4967591
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967477, upper bound: 3.4967591
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967421, upper bound: 3.4967698
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967421, upper bound: 3.4967698
time: 1.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967415, upper bound: 3.4967627
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967415, upper bound: 3.4967627
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967380, upper bound: 3.4967740
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967380, upper bound: 3.4967740
time: 1.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967739, upper bound: 3.4967380
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967739, upper bound: 3.4967380
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967627, upper bound: 3.4967415
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967627, upper bound: 3.4967415
time: 1.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967698, upper bound: 3.4967421
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967698, upper bound: 3.4967421
time: 1.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967591, upper bound: 3.4967477
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967591, upper bound: 3.4967477
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967469, upper bound: 3.4967664
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967469, upper bound: 3.4967664
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967402, upper bound: 3.4967737
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967402, upper bound: 3.4967737
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967401, upper bound: 3.4967684
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967401, upper bound: 3.4967684
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967355, upper bound: 3.4967774
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967355, upper bound: 3.4967774
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967774, upper bound: 3.4967367
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967774, upper bound: 3.4967367
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967689, upper bound: 3.4967414
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967689, upper bound: 3.4967414
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967731, upper bound: 3.4967403
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967731, upper bound: 3.4967403
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967667, upper bound: 3.4967469
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967667, upper bound: 3.4967469
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967475, upper bound: 3.4967591
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967475, upper bound: 3.4967591
time: 1.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967419, upper bound: 3.4967702
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967419, upper bound: 3.4967702
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967406, upper bound: 3.4967627
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967406, upper bound: 3.4967627
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967370, upper bound: 3.4967739
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967370, upper bound: 3.4967739
time: 1.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967739, upper bound: 3.4967370
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967739, upper bound: 3.4967370
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967627, upper bound: 3.4967406
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967627, upper bound: 3.4967406
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967702, upper bound: 3.4967419
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967702, upper bound: 3.4967419
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967591, upper bound: 3.4967475
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967591, upper bound: 3.4967475
time: 1.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967469, upper bound: 3.4967667
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4967469, upper bound: 3.4967667
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398
1: -1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865
2: -1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576
3: -1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979
4: -1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516
5: -1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349
6: -1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487
7: -1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090
8: -2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340
9: -1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 5.12 + 596.52 = 601.64 seconds
