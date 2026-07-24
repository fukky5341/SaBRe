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
execution time: IAR + RelationalAnalysis = 1.33 + 3.65 = 4.98 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -3.7040631, upper bound: 3.7040631

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7038009, upper bound: 3.7038009
time: 2.82 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7038009, upper bound: 3.7038009
time: 1.77 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 4.60 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 4.60
Output dim: 8, lower bound: -3.7038009, upper bound: 3.7038009
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 4.60
Output dim: 8, lower bound: -3.7038009, upper bound: 3.7038009

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7038009, upper bound: 3.7038008
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7038008, upper bound: 3.7038009
time: 1.60 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7036072, upper bound: 3.7035688
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7035688, upper bound: 3.7036072
time: 1.77 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.33 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.33
Output dim: 8, lower bound: -3.7038009, upper bound: 3.7038008
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.33
Output dim: 8, lower bound: -3.7038008, upper bound: 3.7038009
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.33
Output dim: 8, lower bound: -3.7036072, upper bound: 3.7035688
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.33
Output dim: 8, lower bound: -3.7035688, upper bound: 3.7036072

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7035634, upper bound: 3.7035476
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7035470, upper bound: 3.7035633
time: 1.48 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7033811, upper bound: 3.7033808
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7033811, upper bound: 3.7033808
time: 1.46 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7034107, upper bound: 3.7033582
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7034137, upper bound: 3.7033485
time: 1.35 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7033024, upper bound: 3.7033593
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7033024, upper bound: 3.7033593
time: 1.69 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.55 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.55
Output dim: 8, lower bound: -3.7035634, upper bound: 3.7035476
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.55
Output dim: 8, lower bound: -3.7035470, upper bound: 3.7035633
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.55
Output dim: 8, lower bound: -3.7033811, upper bound: 3.7033808
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.55
Output dim: 8, lower bound: -3.7033811, upper bound: 3.7033808
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.55
Output dim: 8, lower bound: -3.7034107, upper bound: 3.7033582
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.55
Output dim: 8, lower bound: -3.7034137, upper bound: 3.7033485
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.55
Output dim: 8, lower bound: -3.7033024, upper bound: 3.7033593
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.55
Output dim: 8, lower bound: -3.7033024, upper bound: 3.7033593

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7032908, upper bound: 3.7032904
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7033101, upper bound: 3.7032677
time: 1.38 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6708781, upper bound: 3.6708871
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6708781, upper bound: 3.6708871
time: 1.38 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 117

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6965765, upper bound: 3.6965958
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6965960, upper bound: 3.6965749
time: 1.29 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6975434, upper bound: 3.6975497
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6975492, upper bound: 3.6975439
time: 1.39 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7028198, upper bound: 3.7027955
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7028198, upper bound: 3.7027955
time: 1.47 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6632107, upper bound: 3.6631606
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6632107, upper bound: 3.6631606
time: 1.44 seconds

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4036629, upper bound: 3.4036629
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4036629, upper bound: 3.4036629
time: 1.23 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7033023, upper bound: 3.7033589
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7033024, upper bound: 3.7033593
time: 1.57 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.53 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 8, lower bound: -3.7032908, upper bound: 3.7032904
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 8, lower bound: -3.7033101, upper bound: 3.7032677
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 8, lower bound: -3.6708781, upper bound: 3.6708871
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 8, lower bound: -3.6708781, upper bound: 3.6708871
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 8, lower bound: -3.6965765, upper bound: 3.6965958
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 8, lower bound: -3.6965960, upper bound: 3.6965749
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 8, lower bound: -3.6975434, upper bound: 3.6975497
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 8, lower bound: -3.6975492, upper bound: 3.6975439
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 8, lower bound: -3.7028198, upper bound: 3.7027955
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 8, lower bound: -3.7028198, upper bound: 3.7027955
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 8, lower bound: -3.6632107, upper bound: 3.6631606
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 8, lower bound: -3.6632107, upper bound: 3.6631606
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.53
Output dim: 8, lower bound: -3.4036629, upper bound: 3.4036629
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.53
Output dim: 8, lower bound: -3.4036629, upper bound: 3.4036629
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 8, lower bound: -3.7033023, upper bound: 3.7033589
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 8, lower bound: -3.7033024, upper bound: 3.7033593

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 117

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6837942, upper bound: 3.6837950
time: 1.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6837942, upper bound: 3.6837950
time: 1.49 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6879553, upper bound: 3.6879830
time: 4.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6880289, upper bound: 3.6879121
time: 1.52 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6708372, upper bound: 3.6707299
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6707392, upper bound: 3.6708381
time: 1.45 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6366814, upper bound: 3.6367131
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6366944, upper bound: 3.6366978
time: 1.20 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6848203, upper bound: 3.6848382
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6848203, upper bound: 3.6848382
time: 1.34 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4282299, upper bound: 3.4281561
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4282299, upper bound: 3.4281561
time: 1.35 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.3812709, upper bound: 3.3811811
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.3812709, upper bound: 3.3811811
time: 1.16 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6799120, upper bound: 3.6799618
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6799658, upper bound: 3.6799032
time: 1.51 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5224211, upper bound: 3.5224383
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5224211, upper bound: 3.5224383
time: 1.28 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6796758, upper bound: 3.6796501
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6796779, upper bound: 3.6796469
time: 1.64 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6632107, upper bound: 3.6631606
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6632107, upper bound: 3.6631606
time: 1.24 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6269755, upper bound: 3.6269700
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6269755, upper bound: 3.6269700
time: 1.23 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.0777451, upper bound: 3.0777657
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.0777451, upper bound: 3.0777657
time: 0.87 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6651143, upper bound: 3.6651403
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6651143, upper bound: 3.6651403
time: 1.58 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.43 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 8, lower bound: -3.6837942, upper bound: 3.6837950
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 8, lower bound: -3.6837942, upper bound: 3.6837950
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 8, lower bound: -3.6879553, upper bound: 3.6879830
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 8, lower bound: -3.6880289, upper bound: 3.6879121
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 8, lower bound: -3.6708372, upper bound: 3.6707299
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 8, lower bound: -3.6707392, upper bound: 3.6708381
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 8, lower bound: -3.6366814, upper bound: 3.6367131
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 8, lower bound: -3.6366944, upper bound: 3.6366978
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 8, lower bound: -3.6848203, upper bound: 3.6848382
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 8, lower bound: -3.6848203, upper bound: 3.6848382
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.43
Output dim: 8, lower bound: -3.4282299, upper bound: 3.4281561
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.43
Output dim: 8, lower bound: -3.4282299, upper bound: 3.4281561
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.43
Output dim: 8, lower bound: -3.3812709, upper bound: 3.3811811
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.43
Output dim: 8, lower bound: -3.3812709, upper bound: 3.3811811
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 8, lower bound: -3.6799120, upper bound: 3.6799618
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 8, lower bound: -3.6799658, upper bound: 3.6799032
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 8, lower bound: -3.5224211, upper bound: 3.5224383
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 8, lower bound: -3.5224211, upper bound: 3.5224383
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 8, lower bound: -3.6796758, upper bound: 3.6796501
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 8, lower bound: -3.6796779, upper bound: 3.6796469
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 8, lower bound: -3.6632107, upper bound: 3.6631606
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 8, lower bound: -3.6632107, upper bound: 3.6631606
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 8, lower bound: -3.6269755, upper bound: 3.6269700
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 8, lower bound: -3.6269755, upper bound: 3.6269700
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.43
Output dim: 8, lower bound: -3.0777451, upper bound: 3.0777657
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.43
Output dim: 8, lower bound: -3.0777451, upper bound: 3.0777657
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 8, lower bound: -3.6651143, upper bound: 3.6651403
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 8, lower bound: -3.6651143, upper bound: 3.6651403

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6593711, upper bound: 3.6594074
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6593909, upper bound: 3.6593828
time: 1.28 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6836429, upper bound: 3.6836412
time: 2.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6836430, upper bound: 3.6836410
time: 1.31 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.9570783, upper bound: 2.9570783
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.9570783, upper bound: 2.9570783
time: 0.98 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6626363, upper bound: 3.6625462
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6626499, upper bound: 3.6625275
time: 1.61 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6366711, upper bound: 3.6365527
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6366831, upper bound: 3.6365325
time: 1.29 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6264387, upper bound: 3.6265234
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6264385, upper bound: 3.6265234
time: 1.48 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.3074378, upper bound: 3.3076209
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.3074378, upper bound: 3.3076209
time: 1.14 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6121122, upper bound: 3.6121086
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6121118, upper bound: 3.6121086
time: 1.20 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6590878, upper bound: 3.6590883
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6590699, upper bound: 3.6591051
time: 1.26 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 117

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.3968538, upper bound: 3.3968457
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.3968538, upper bound: 3.3968457
time: 1.04 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6643217, upper bound: 3.6643958
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6643406, upper bound: 3.6643686
time: 1.58 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6611672, upper bound: 3.6611135
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6611826, upper bound: 3.6610963
time: 1.23 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5224211, upper bound: 3.5223912
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5223759, upper bound: 3.5224383
time: 1.18 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5224210, upper bound: 3.5223319
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5223282, upper bound: 3.5224383
time: 1.15 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6472184, upper bound: 3.6472325
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6472184, upper bound: 3.6472325
time: 1.39 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6549941, upper bound: 3.6549656
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6549941, upper bound: 3.6549656
time: 1.41 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6283418, upper bound: 3.6283335
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6283418, upper bound: 3.6283335
time: 1.19 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6630366, upper bound: 3.6629930
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6630387, upper bound: 3.6629931
time: 1.42 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.1731298, upper bound: 3.1731586
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.1731298, upper bound: 3.1731586
time: 1.13 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.2880047, upper bound: 3.2880200
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.2880047, upper bound: 3.2880200
time: 1.05 seconds

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6426710, upper bound: 3.6427087
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6426816, upper bound: 3.6427012
time: 1.34 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5716682, upper bound: 3.5716493
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5716682, upper bound: 3.5716493
time: 1.27 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.80 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.6593711, upper bound: 3.6594074
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.6593909, upper bound: 3.6593828
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.6836429, upper bound: 3.6836412
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.6836430, upper bound: 3.6836410
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.80
Output dim: 8, lower bound: -2.9570783, upper bound: 2.9570783
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.80
Output dim: 8, lower bound: -2.9570783, upper bound: 2.9570783
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.6626363, upper bound: 3.6625462
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.6626499, upper bound: 3.6625275
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.6366711, upper bound: 3.6365527
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.6366831, upper bound: 3.6365325
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.6264387, upper bound: 3.6265234
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.6264385, upper bound: 3.6265234
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.3074378, upper bound: 3.3076209
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.3074378, upper bound: 3.3076209
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.6121122, upper bound: 3.6121086
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.6121118, upper bound: 3.6121086
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.6590878, upper bound: 3.6590883
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.6590699, upper bound: 3.6591051
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.3968538, upper bound: 3.3968457
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.3968538, upper bound: 3.3968457
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.6643217, upper bound: 3.6643958
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.6643406, upper bound: 3.6643686
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.6611672, upper bound: 3.6611135
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.6611826, upper bound: 3.6610963
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.5224211, upper bound: 3.5223912
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.5223759, upper bound: 3.5224383
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.5224210, upper bound: 3.5223319
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.5223282, upper bound: 3.5224383
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.6472184, upper bound: 3.6472325
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.6472184, upper bound: 3.6472325
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.6549941, upper bound: 3.6549656
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.6549941, upper bound: 3.6549656
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.6283418, upper bound: 3.6283335
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.6283418, upper bound: 3.6283335
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.6630366, upper bound: 3.6629930
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.6630387, upper bound: 3.6629931
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.1731298, upper bound: 3.1731586
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.1731298, upper bound: 3.1731586
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.2880047, upper bound: 3.2880200
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.2880047, upper bound: 3.2880200
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.6426710, upper bound: 3.6427087
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.6426816, upper bound: 3.6427012
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.5716682, upper bound: 3.5716493
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -3.5716682, upper bound: 3.5716493

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5557111, upper bound: 3.5558173
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5557111, upper bound: 3.5558173
time: 1.26 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6592356, upper bound: 3.6592131
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6592368, upper bound: 3.6592078
time: 1.72 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.0000844, upper bound: 3.0000844
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.0000844, upper bound: 3.0000844
time: 1.07 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6830089, upper bound: 3.6829735
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6829545, upper bound: 3.6830085
time: 1.45 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5532264, upper bound: 3.5530604
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5532264, upper bound: 3.5530604
time: 1.22 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6285764, upper bound: 3.6284757
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6285764, upper bound: 3.6284757
time: 1.23 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 117

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.1855165, upper bound: 3.1856277
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.1855165, upper bound: 3.1856277
time: 1.04 seconds

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6121117, upper bound: 3.6119653
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6121115, upper bound: 3.6119653
time: 1.33 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6264387, upper bound: 3.6264799
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6264054, upper bound: 3.6265234
time: 1.59 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6251713, upper bound: 3.6252853
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6251649, upper bound: 3.6252956
time: 1.52 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5946180, upper bound: 3.5946842
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5946867, upper bound: 3.5946055
time: 1.33 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5607439, upper bound: 3.5607722
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5607426, upper bound: 3.5607722
time: 1.19 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5548260, upper bound: 3.5547776
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5548260, upper bound: 3.5547776
time: 1.18 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6590699, upper bound: 3.6590253
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6590038, upper bound: 3.6591050
time: 1.69 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6643217, upper bound: 3.6643294
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6642745, upper bound: 3.6643958
time: 1.23 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.2930380, upper bound: 3.2929295
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.2930380, upper bound: 3.2929295
time: 1.15 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5825836, upper bound: 3.5825494
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5825836, upper bound: 3.5825494
time: 1.32 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.3512802, upper bound: 3.3510780
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.3512802, upper bound: 3.3510780
time: 1.00 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5224172, upper bound: 3.5223911
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5224211, upper bound: 3.5223845
time: 1.46 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4984286, upper bound: 3.4985696
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4985086, upper bound: 3.4985178
time: 1.34 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5222205, upper bound: 3.5221255
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5222191, upper bound: 3.5221300
time: 1.39 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.1309237, upper bound: 3.1309228
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.1309237, upper bound: 3.1309228
time: 0.94 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5932957, upper bound: 3.5933054
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5932957, upper bound: 3.5933054
time: 1.30 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6302271, upper bound: 3.6302630
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6302597, upper bound: 3.6302289
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6134276, upper bound: 3.6134767
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6134940, upper bound: 3.6134159
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.5160520, upper bound: 3.5160281
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.5160520, upper bound: 3.5160281
time: 1.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6282008, upper bound: 3.6281922
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6282003, upper bound: 3.6281929
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.2956096, upper bound: 3.2957120
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.2956096, upper bound: 3.2957120
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6379314, upper bound: 3.6378732
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6379197, upper bound: 3.6378780
time: 1.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.2984077, upper bound: 3.2983974
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.2984077, upper bound: 3.2983974
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5985041, upper bound: 3.5985310
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5985106, upper bound: 3.5985299
time: 1.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5985087, upper bound: 3.5985294
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5985178, upper bound: 3.5985284
time: 1.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5715819, upper bound: 3.5715963
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5715986, upper bound: 3.5715848
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5508875, upper bound: 3.5509123
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5509347, upper bound: 3.5508686
time: 1.16 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.89 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.5557111, upper bound: 3.5558173
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.5557111, upper bound: 3.5558173
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.6592356, upper bound: 3.6592131
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.6592368, upper bound: 3.6592078
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.0000844, upper bound: 3.0000844
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.0000844, upper bound: 3.0000844
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.6830089, upper bound: 3.6829735
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.6829545, upper bound: 3.6830085
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.5532264, upper bound: 3.5530604
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.5532264, upper bound: 3.5530604
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.6285764, upper bound: 3.6284757
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.6285764, upper bound: 3.6284757
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.1855165, upper bound: 3.1856277
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.1855165, upper bound: 3.1856277
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.6121117, upper bound: 3.6119653
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.6121115, upper bound: 3.6119653
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.6264387, upper bound: 3.6264799
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.6264054, upper bound: 3.6265234
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.6251713, upper bound: 3.6252853
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.6251649, upper bound: 3.6252956
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.5946180, upper bound: 3.5946842
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.5946867, upper bound: 3.5946055
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.5607439, upper bound: 3.5607722
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.5607426, upper bound: 3.5607722
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.5548260, upper bound: 3.5547776
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.5548260, upper bound: 3.5547776
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.6590699, upper bound: 3.6590253
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.6590038, upper bound: 3.6591050
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.6643217, upper bound: 3.6643294
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.6642745, upper bound: 3.6643958
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.2930380, upper bound: 3.2929295
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.2930380, upper bound: 3.2929295
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.5825836, upper bound: 3.5825494
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.5825836, upper bound: 3.5825494
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.3512802, upper bound: 3.3510780
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.3512802, upper bound: 3.3510780
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.5224172, upper bound: 3.5223911
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.5224211, upper bound: 3.5223845
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.4984286, upper bound: 3.4985696
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.4985086, upper bound: 3.4985178
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.5222205, upper bound: 3.5221255
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.5222191, upper bound: 3.5221300
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.1309237, upper bound: 3.1309228
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.1309237, upper bound: 3.1309228
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.5932957, upper bound: 3.5933054
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.5932957, upper bound: 3.5933054
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.6302271, upper bound: 3.6302630
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.6302597, upper bound: 3.6302289
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.6134276, upper bound: 3.6134767
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.6134940, upper bound: 3.6134159
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.5160520, upper bound: 3.5160281
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.5160520, upper bound: 3.5160281
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.6282008, upper bound: 3.6281922
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.6282003, upper bound: 3.6281929
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.2956096, upper bound: 3.2957120
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.2956096, upper bound: 3.2957120
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.6379314, upper bound: 3.6378732
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.6379197, upper bound: 3.6378780
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.2984077, upper bound: 3.2983974
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.2984077, upper bound: 3.2983974
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.5985041, upper bound: 3.5985310
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.5985106, upper bound: 3.5985299
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.5985087, upper bound: 3.5985294
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.5985178, upper bound: 3.5985284
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.5715819, upper bound: 3.5715963
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.5715986, upper bound: 3.5715848
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.5508875, upper bound: 3.5509123
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.89
Output dim: 8, lower bound: -3.5509347, upper bound: 3.5508686

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.3703636, upper bound: 3.3706055
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.3703636, upper bound: 3.3706055
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5214999, upper bound: 3.5216006
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5215014, upper bound: 3.5215997
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.3558040, upper bound: 3.3558840
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.3558040, upper bound: 3.3558840
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6167752, upper bound: 3.6167282
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6167772, upper bound: 3.6167226
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.3893434, upper bound: 3.3894296
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.3893434, upper bound: 3.3894296
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6620269, upper bound: 3.6620787
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6620444, upper bound: 3.6620778
time: 1.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4717290, upper bound: 3.4714943
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4717290, upper bound: 3.4714943
time: 1.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5532264, upper bound: 3.5529291
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5531240, upper bound: 3.5530604
time: 1.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.3268454, upper bound: 3.3266057
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.3268454, upper bound: 3.3266057
time: 1.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6284034, upper bound: 3.6283102
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6284028, upper bound: 3.6283149
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5941142, upper bound: 3.5940095
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5941142, upper bound: 3.5940095
time: 1.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5941069, upper bound: 3.5940096
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5941069, upper bound: 3.5940096
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6065726, upper bound: 3.6065876
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6065726, upper bound: 3.6065876
time: 1.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6251334, upper bound: 3.6252853
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6251172, upper bound: 3.6252956
time: 1.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5263632, upper bound: 3.5264856
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5263632, upper bound: 3.5264856
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5984981, upper bound: 3.5986227
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5985053, upper bound: 3.5986165
time: 1.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5946180, upper bound: 3.5946252
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5945783, upper bound: 3.5946842
time: 1.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.3747065, upper bound: 3.3745933
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.3747065, upper bound: 3.3745933
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5431850, upper bound: 3.5431868
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5431850, upper bound: 3.5431868
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5431850, upper bound: 3.5431868
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5431850, upper bound: 3.5431868
time: 1.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5340524, upper bound: 3.5341034
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5341315, upper bound: 3.5340500
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 117

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5548203, upper bound: 3.5547729
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5548214, upper bound: 3.5547726
time: 1.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6030652, upper bound: 3.6030146
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6030652, upper bound: 3.6030146
time: 1.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6427446, upper bound: 3.6429613
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6428418, upper bound: 3.6429090
time: 1.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.5076510, upper bound: 3.5076768
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.5076510, upper bound: 3.5076768
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5863182, upper bound: 3.5864826
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5863182, upper bound: 3.5864826
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5538470, upper bound: 3.5538386
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5538470, upper bound: 3.5538386
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.1815228, upper bound: 3.1816553
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.1815228, upper bound: 3.1816553
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.1309104, upper bound: 3.1309192
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.1309104, upper bound: 3.1309192
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4843427, upper bound: 3.4842947
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4843427, upper bound: 3.4842947
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4746537, upper bound: 3.4746256
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4747020, upper bound: 3.4745777
time: 1.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4842283, upper bound: 3.4841232
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4842283, upper bound: 3.4841232
time: 1.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.3545053, upper bound: 3.3545651
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.3545053, upper bound: 3.3545651
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5423497, upper bound: 3.5423713
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5423497, upper bound: 3.5423713
time: 1.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6302221, upper bound: 3.6301459
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6301350, upper bound: 3.6302415
time: 1.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6073251, upper bound: 3.6073149
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6073251, upper bound: 3.6073149
time: 1.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5464815, upper bound: 3.5464985
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5464815, upper bound: 3.5464985
time: 1.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5305131, upper bound: 3.5304731
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5305131, upper bound: 3.5304731
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4486893, upper bound: 3.4486985
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4486893, upper bound: 3.4486985
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6042476, upper bound: 3.6042486
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6042475, upper bound: 3.6042486
time: 1.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5984812, upper bound: 3.5984548
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5984812, upper bound: 3.5984548
time: 1.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.2322745, upper bound: 3.2322270
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.2322745, upper bound: 3.2322270
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5636704, upper bound: 3.5637777
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5637501, upper bound: 3.5636919
time: 1.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5771809, upper bound: 3.5771567
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5771809, upper bound: 3.5771573
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5985087, upper bound: 3.5984102
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5984110, upper bound: 3.5985294
time: 1.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4726026, upper bound: 3.4725934
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4726026, upper bound: 3.4725934
time: 1.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5715819, upper bound: 3.5714686
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5714749, upper bound: 3.5715963
time: 1.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.3799925, upper bound: 3.3799712
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.3799925, upper bound: 3.3799712
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.1168811, upper bound: 3.1170170
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.1168811, upper bound: 3.1170170
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.0808125, upper bound: 3.0807692
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.0808125, upper bound: 3.0807692
time: 1.02 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.42 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.3703636, upper bound: 3.3706055
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.3703636, upper bound: 3.3706055
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5214999, upper bound: 3.5216006
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5215014, upper bound: 3.5215997
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.3558040, upper bound: 3.3558840
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.3558040, upper bound: 3.3558840
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.6167752, upper bound: 3.6167282
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.6167772, upper bound: 3.6167226
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.3893434, upper bound: 3.3894296
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.3893434, upper bound: 3.3894296
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.6620269, upper bound: 3.6620787
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.6620444, upper bound: 3.6620778
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.4717290, upper bound: 3.4714943
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.4717290, upper bound: 3.4714943
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5532264, upper bound: 3.5529291
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5531240, upper bound: 3.5530604
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.3268454, upper bound: 3.3266057
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.3268454, upper bound: 3.3266057
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.6284034, upper bound: 3.6283102
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.6284028, upper bound: 3.6283149
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5941142, upper bound: 3.5940095
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5941142, upper bound: 3.5940095
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5941069, upper bound: 3.5940096
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5941069, upper bound: 3.5940096
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.6065726, upper bound: 3.6065876
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.6065726, upper bound: 3.6065876
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.6251334, upper bound: 3.6252853
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.6251172, upper bound: 3.6252956
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5263632, upper bound: 3.5264856
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5263632, upper bound: 3.5264856
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5984981, upper bound: 3.5986227
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5985053, upper bound: 3.5986165
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5946180, upper bound: 3.5946252
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5945783, upper bound: 3.5946842
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.3747065, upper bound: 3.3745933
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.3747065, upper bound: 3.3745933
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5431850, upper bound: 3.5431868
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5431850, upper bound: 3.5431868
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5431850, upper bound: 3.5431868
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5431850, upper bound: 3.5431868
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5340524, upper bound: 3.5341034
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5341315, upper bound: 3.5340500
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5548203, upper bound: 3.5547729
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5548214, upper bound: 3.5547726
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.6030652, upper bound: 3.6030146
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.6030652, upper bound: 3.6030146
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.6427446, upper bound: 3.6429613
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.6428418, upper bound: 3.6429090
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5076510, upper bound: 3.5076768
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5076510, upper bound: 3.5076768
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5863182, upper bound: 3.5864826
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5863182, upper bound: 3.5864826
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5538470, upper bound: 3.5538386
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5538470, upper bound: 3.5538386
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.1815228, upper bound: 3.1816553
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.1815228, upper bound: 3.1816553
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.1309104, upper bound: 3.1309192
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.1309104, upper bound: 3.1309192
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.4843427, upper bound: 3.4842947
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.4843427, upper bound: 3.4842947
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.4746537, upper bound: 3.4746256
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.4747020, upper bound: 3.4745777
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.4842283, upper bound: 3.4841232
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.4842283, upper bound: 3.4841232
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.3545053, upper bound: 3.3545651
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.3545053, upper bound: 3.3545651
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5423497, upper bound: 3.5423713
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5423497, upper bound: 3.5423713
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.6302221, upper bound: 3.6301459
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.6301350, upper bound: 3.6302415
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.6073251, upper bound: 3.6073149
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.6073251, upper bound: 3.6073149
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5464815, upper bound: 3.5464985
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5464815, upper bound: 3.5464985
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5305131, upper bound: 3.5304731
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5305131, upper bound: 3.5304731
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.4486893, upper bound: 3.4486985
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.4486893, upper bound: 3.4486985
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.6042476, upper bound: 3.6042486
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.6042475, upper bound: 3.6042486
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5984812, upper bound: 3.5984548
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5984812, upper bound: 3.5984548
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.2322745, upper bound: 3.2322270
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.2322745, upper bound: 3.2322270
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5636704, upper bound: 3.5637777
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5637501, upper bound: 3.5636919
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5771809, upper bound: 3.5771567
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5771809, upper bound: 3.5771573
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5985087, upper bound: 3.5984102
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5984110, upper bound: 3.5985294
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.4726026, upper bound: 3.4725934
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.4726026, upper bound: 3.4725934
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5715819, upper bound: 3.5714686
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.5714749, upper bound: 3.5715963
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.3799925, upper bound: 3.3799712
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.3799925, upper bound: 3.3799712
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.1168811, upper bound: 3.1170170
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.1168811, upper bound: 3.1170170
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.0808125, upper bound: 3.0807692
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.42
Output dim: 8, lower bound: -3.0808125, upper bound: 3.0807692

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.3322420, upper bound: 3.3324252
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.3322420, upper bound: 3.3324252
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.1889598, upper bound: 3.1891542
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.1889598, upper bound: 3.1891542
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4094074, upper bound: 3.4094429
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4094074, upper bound: 3.4094429
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4602898, upper bound: 3.4601938
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4602898, upper bound: 3.4601938
time: 1.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6268983, upper bound: 3.6269564
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6268983, upper bound: 3.6269564
time: 1.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.5046499, upper bound: 3.5046449
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.5046499, upper bound: 3.5046449
time: 1.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.1937911, upper bound: 3.1936739
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.1937911, upper bound: 3.1936739
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4716455, upper bound: 3.4714943
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4716455, upper bound: 3.4714943
time: 1.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5938617, upper bound: 3.5937599
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5938617, upper bound: 3.5937599
time: 1.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.2575251, upper bound: 3.2573246
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.2575251, upper bound: 3.2573246
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5621463, upper bound: 3.5620887
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5621477, upper bound: 3.5620884
time: 1.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5621463, upper bound: 3.5620887
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5621477, upper bound: 3.5620884
time: 1.28 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 3.94 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.94
Output dim: 8, lower bound: -3.3322420, upper bound: 3.3324252
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.94
Output dim: 8, lower bound: -3.3322420, upper bound: 3.3324252
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.94
Output dim: 8, lower bound: -3.1889598, upper bound: 3.1891542
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.94
Output dim: 8, lower bound: -3.1889598, upper bound: 3.1891542
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.94
Output dim: 8, lower bound: -3.4094074, upper bound: 3.4094429
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.94
Output dim: 8, lower bound: -3.4094074, upper bound: 3.4094429
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.94
Output dim: 8, lower bound: -3.4602898, upper bound: 3.4601938
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.94
Output dim: 8, lower bound: -3.4602898, upper bound: 3.4601938
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.94
Output dim: 8, lower bound: -3.6268983, upper bound: 3.6269564
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.94
Output dim: 8, lower bound: -3.6268983, upper bound: 3.6269564
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.94
Output dim: 8, lower bound: -3.5046499, upper bound: 3.5046449
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.94
Output dim: 8, lower bound: -3.5046499, upper bound: 3.5046449
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.94
Output dim: 8, lower bound: -3.1937911, upper bound: 3.1936739
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.94
Output dim: 8, lower bound: -3.1937911, upper bound: 3.1936739
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.94
Output dim: 8, lower bound: -3.4716455, upper bound: 3.4714943
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.94
Output dim: 8, lower bound: -3.4716455, upper bound: 3.4714943
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.94
Output dim: 8, lower bound: -3.5938617, upper bound: 3.5937599
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.94
Output dim: 8, lower bound: -3.5938617, upper bound: 3.5937599
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.94
Output dim: 8, lower bound: -3.2575251, upper bound: 3.2573246
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.94
Output dim: 8, lower bound: -3.2575251, upper bound: 3.2573246
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.94
Output dim: 8, lower bound: -3.5621463, upper bound: 3.5620887
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.94
Output dim: 8, lower bound: -3.5621477, upper bound: 3.5620884
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.94
Output dim: 8, lower bound: -3.5621463, upper bound: 3.5620887
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.94
Output dim: 8, lower bound: -3.5621477, upper bound: 3.5620884
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.5941069, upper bound: 3.5940096
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.5941069, upper bound: 3.5940096
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.6065726, upper bound: 3.6065876
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.6065726, upper bound: 3.6065876
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.6251334, upper bound: 3.6252853
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.6251172, upper bound: 3.6252956
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.5263632, upper bound: 3.5264856
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.5263632, upper bound: 3.5264856
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.5984981, upper bound: 3.5986227
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.5985053, upper bound: 3.5986165
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.5946180, upper bound: 3.5946252
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.5945783, upper bound: 3.5946842
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.5431850, upper bound: 3.5431868
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.5431850, upper bound: 3.5431868
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.5431850, upper bound: 3.5431868
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.5431850, upper bound: 3.5431868
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.5340524, upper bound: 3.5341034
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.5341315, upper bound: 3.5340500
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.5548203, upper bound: 3.5547729
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.5548214, upper bound: 3.5547726
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.6030652, upper bound: 3.6030146
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.6030652, upper bound: 3.6030146
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.6427446, upper bound: 3.6429613
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.6428418, upper bound: 3.6429090
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.5863182, upper bound: 3.5864826
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.5863182, upper bound: 3.5864826
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.5538470, upper bound: 3.5538386
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.5538470, upper bound: 3.5538386
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.5423497, upper bound: 3.5423713
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.5423497, upper bound: 3.5423713
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.6302221, upper bound: 3.6301459
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.6301350, upper bound: 3.6302415
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.6073251, upper bound: 3.6073149
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.6073251, upper bound: 3.6073149
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.5464815, upper bound: 3.5464985
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.5464815, upper bound: 3.5464985
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.5305131, upper bound: 3.5304731
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.5305131, upper bound: 3.5304731
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.6042476, upper bound: 3.6042486
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.6042475, upper bound: 3.6042486
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.5984812, upper bound: 3.5984548
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.5984812, upper bound: 3.5984548
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.5636704, upper bound: 3.5637777
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.5637501, upper bound: 3.5636919
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.5771809, upper bound: 3.5771567
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.5771809, upper bound: 3.5771573
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.5985087, upper bound: 3.5984102
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.5984110, upper bound: 3.5985294
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.5715819, upper bound: 3.5714686
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -3.5714749, upper bound: 3.5715963

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 4.98 + 596.82 = 601.81 seconds
