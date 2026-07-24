## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00014742


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001197, 0.0001197)
1: (-0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002319, 0.0002319)
2: (0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0018704, 0.0018704)
3: (-0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001671, 0.0001671)
4: (0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0008105, 0.0008105)
5: (-0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001210, 0.0001210)
6: (0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0002219, 0.0002219)
7: (-0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0014671, 0.0014671)
8: (-0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004596, 0.0004596)
9: (-0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0009174, 0.0009174)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.55 + 1.44 = 2.99 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0001740, upper bound: 0.0001741

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001686, upper bound: 0.0001679
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001679, upper bound: 0.0001687
time: 0.53 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.22 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.22
Output dim: 6, lower bound: -0.0001686, upper bound: 0.0001679
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.22
Output dim: 6, lower bound: -0.0001679, upper bound: 0.0001687

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001179, 0.0001175
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002283, 0.0002276
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0018357, 0.0018415
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001645, 0.0001640
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007980, 0.0007955
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001187, 0.0001191
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0002185, 0.0002178
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0014445, 0.0014399
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004525, 0.0004511
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0009004, 0.0009032

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001664, upper bound: 0.0001660
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001668, upper bound: 0.0001651
time: 0.55 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001175, 0.0001197
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002276, 0.0002319
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0018704, 0.0018357
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001640, 0.0001671
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007955, 0.0008105
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001210, 0.0001187
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0002178, 0.0002219
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0014399, 0.0014671
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004511, 0.0004596
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0009174, 0.0009004

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001651, upper bound: 0.0001668
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001661, upper bound: 0.0001664
time: 0.54 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.62 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.62
Output dim: 6, lower bound: -0.0001664, upper bound: 0.0001660
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.62
Output dim: 6, lower bound: -0.0001668, upper bound: 0.0001651
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.62
Output dim: 6, lower bound: -0.0001651, upper bound: 0.0001668
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.62
Output dim: 6, lower bound: -0.0001661, upper bound: 0.0001664

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001177, 0.0001176
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002279, 0.0002277
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0018369, 0.0018383
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001642, 0.0001641
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007966, 0.0007960
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001188, 0.0001189
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0002181, 0.0002179
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0014420, 0.0014409
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004518, 0.0004514
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0009010, 0.0009017

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001643, upper bound: 0.0001578
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001558, upper bound: 0.0001634
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001179, 0.0001173
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002285, 0.0002272
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0018323, 0.0018427
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001646, 0.0001637
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007985, 0.0007940
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001185, 0.0001192
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0002186, 0.0002174
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0014454, 0.0014373
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004528, 0.0004503
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008987, 0.0009038

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001645, upper bound: 0.0001556
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001581, upper bound: 0.0001628
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001173, 0.0001197
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002272, 0.0002319
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0018705, 0.0018323
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001637, 0.0001671
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007940, 0.0008106
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001210, 0.0001185
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0002174, 0.0002219
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0014373, 0.0014673
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004503, 0.0004597
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0009175, 0.0008987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001628, upper bound: 0.0001581
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001556, upper bound: 0.0001645
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001176, 0.0001194
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002277, 0.0002313
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0018660, 0.0018369
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001641, 0.0001667
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007960, 0.0008086
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001207, 0.0001188
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0002179, 0.0002214
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0014409, 0.0014637
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004514, 0.0004586
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0009152, 0.0009010

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001633, upper bound: 0.0001559
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001578, upper bound: 0.0001643
time: 0.57 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 8.27 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 8.27
Output dim: 6, lower bound: -0.0001643, upper bound: 0.0001578
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 8.27
Output dim: 6, lower bound: -0.0001558, upper bound: 0.0001634
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 8.27
Output dim: 6, lower bound: -0.0001645, upper bound: 0.0001556
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 8.27
Output dim: 6, lower bound: -0.0001581, upper bound: 0.0001628
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 8.27
Output dim: 6, lower bound: -0.0001628, upper bound: 0.0001581
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 8.27
Output dim: 6, lower bound: -0.0001556, upper bound: 0.0001645
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 8.27
Output dim: 6, lower bound: -0.0001633, upper bound: 0.0001559
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 8.27
Output dim: 6, lower bound: -0.0001578, upper bound: 0.0001643

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001160, 0.0001145
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002247, 0.0002217
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0017883, 0.0018122
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001619, 0.0001597
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007853, 0.0007749
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001157, 0.0001172
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0002150, 0.0002122
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0014215, 0.0014027
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004453, 0.0004395
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008771, 0.0008888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001625, upper bound: 0.0001510
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001569, upper bound: 0.0001557
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001146, 0.0001157
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002219, 0.0002240
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0018071, 0.0017897
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001598, 0.0001614
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007756, 0.0007831
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001169, 0.0001158
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0002123, 0.0002144
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0014039, 0.0014175
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004398, 0.0004441
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008864, 0.0008778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001536, upper bound: 0.0001557
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001494, upper bound: 0.0001615
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001162, 0.0001142
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002251, 0.0002211
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0017837, 0.0018155
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001621, 0.0001593
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007867, 0.0007730
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001154, 0.0001174
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0002154, 0.0002116
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0014241, 0.0013992
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004462, 0.0004384
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008749, 0.0008905

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001626, upper bound: 0.0001483
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001570, upper bound: 0.0001534
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001148, 0.0001156
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002224, 0.0002238
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0018054, 0.0017941
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001602, 0.0001612
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007774, 0.0007823
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001168, 0.0001161
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0002129, 0.0002142
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0014073, 0.0014162
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004409, 0.0004437
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008855, 0.0008800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001562, upper bound: 0.0001549
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001512, upper bound: 0.0001609
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001156, 0.0001165
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002238, 0.0002256
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0018194, 0.0018054
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001612, 0.0001625
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007823, 0.0007884
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001177, 0.0001168
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0002142, 0.0002159
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0014162, 0.0014271
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004437, 0.0004471
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008924, 0.0008855

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001609, upper bound: 0.0001512
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001549, upper bound: 0.0001561
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001142, 0.0001177
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002211, 0.0002279
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0018382, 0.0017837
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001593, 0.0001642
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007730, 0.0007966
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001189, 0.0001154
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0002116, 0.0002181
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013992, 0.0014419
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004384, 0.0004517
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0009016, 0.0008749

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001534, upper bound: 0.0001570
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001483, upper bound: 0.0001627
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001157, 0.0001162
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002240, 0.0002250
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0018148, 0.0018071
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001614, 0.0001621
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007831, 0.0007864
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001174, 0.0001169
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0002144, 0.0002153
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0014175, 0.0014236
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004441, 0.0004460
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008901, 0.0008864

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001615, upper bound: 0.0001494
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001557, upper bound: 0.0001536
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001145, 0.0001175
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002217, 0.0002277
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0018365, 0.0017883
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001597, 0.0001640
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007749, 0.0007958
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001188, 0.0001157
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0002122, 0.0002179
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0014027, 0.0014406
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004395, 0.0004513
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0009008, 0.0008771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001557, upper bound: 0.0001569
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001510, upper bound: 0.0001625
time: 0.59 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 8.42 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.42
Output dim: 6, lower bound: -0.0001625, upper bound: 0.0001510
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.42
Output dim: 6, lower bound: -0.0001569, upper bound: 0.0001557
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.42
Output dim: 6, lower bound: -0.0001536, upper bound: 0.0001557
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.42
Output dim: 6, lower bound: -0.0001494, upper bound: 0.0001615
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.42
Output dim: 6, lower bound: -0.0001626, upper bound: 0.0001483
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.42
Output dim: 6, lower bound: -0.0001570, upper bound: 0.0001534
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.42
Output dim: 6, lower bound: -0.0001562, upper bound: 0.0001549
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.42
Output dim: 6, lower bound: -0.0001512, upper bound: 0.0001609
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.42
Output dim: 6, lower bound: -0.0001609, upper bound: 0.0001512
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.42
Output dim: 6, lower bound: -0.0001549, upper bound: 0.0001561
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.42
Output dim: 6, lower bound: -0.0001534, upper bound: 0.0001570
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.42
Output dim: 6, lower bound: -0.0001483, upper bound: 0.0001627
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.42
Output dim: 6, lower bound: -0.0001615, upper bound: 0.0001494
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.42
Output dim: 6, lower bound: -0.0001557, upper bound: 0.0001536
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.42
Output dim: 6, lower bound: -0.0001557, upper bound: 0.0001569
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.42
Output dim: 6, lower bound: -0.0001510, upper bound: 0.0001625

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001097, 0.0001077
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002125, 0.0002086
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0016829, 0.0017138
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001531, 0.0001503
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007427, 0.0007293
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001089, 0.0001109
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0002033, 0.0001997
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013443, 0.0013201
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004212, 0.0004136
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008255, 0.0008406

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001606, upper bound: 0.0001484
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001580, upper bound: 0.0001482
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001093, 0.0001084
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002116, 0.0002099
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0016929, 0.0017068
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001524, 0.0001512
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007396, 0.0007336
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001095, 0.0001104
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0002025, 0.0002009
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013389, 0.0013280
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004195, 0.0004160
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008304, 0.0008372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001547, upper bound: 0.0001531
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001540, upper bound: 0.0001534
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001084, 0.0001089
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002099, 0.0002110
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0017018, 0.0016933
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001512, 0.0001520
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007338, 0.0007374
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001101, 0.0001095
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0002009, 0.0002019
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013283, 0.0013349
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004161, 0.0004182
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008347, 0.0008305

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001511, upper bound: 0.0001534
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001512, upper bound: 0.0001534
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001078, 0.0001093
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002088, 0.0002117
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0017076, 0.0016844
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001504, 0.0001525
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007299, 0.0007400
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001105, 0.0001090
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0001998, 0.0002026
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013213, 0.0013395
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004139, 0.0004197
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008376, 0.0008262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001465, upper bound: 0.0001579
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001469, upper bound: 0.0001595
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001099, 0.0001074
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002129, 0.0002081
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0016784, 0.0017172
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001534, 0.0001499
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007441, 0.0007273
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001086, 0.0001111
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0002037, 0.0001991
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013470, 0.0013166
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004220, 0.0004125
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008232, 0.0008423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001607, upper bound: 0.0001457
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001586, upper bound: 0.0001454
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001095, 0.0001081
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002120, 0.0002094
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0016888, 0.0017101
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001527, 0.0001508
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007411, 0.0007318
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001092, 0.0001106
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0002029, 0.0002004
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013415, 0.0013247
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004203, 0.0004150
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008283, 0.0008388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001549, upper bound: 0.0001509
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001542, upper bound: 0.0001508
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001088, 0.0001088
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002107, 0.0002108
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0017000, 0.0016992
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001518, 0.0001518
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007363, 0.0007367
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001100, 0.0001099
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0002016, 0.0002017
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013329, 0.0013335
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004176, 0.0004178
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008339, 0.0008334

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001540, upper bound: 0.0001526
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001532, upper bound: 0.0001526
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001081, 0.0001092
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002094, 0.0002115
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0017063, 0.0016888
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001508, 0.0001524
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007318, 0.0007394
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001104, 0.0001092
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0002004, 0.0002024
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013247, 0.0013384
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004150, 0.0004193
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008369, 0.0008283

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001486, upper bound: 0.0001573
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001487, upper bound: 0.0001590
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001092, 0.0001097
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002115, 0.0002124
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0017133, 0.0017063
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001524, 0.0001530
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007394, 0.0007424
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001108, 0.0001104
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0002024, 0.0002033
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013384, 0.0013439
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004193, 0.0004211
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008404, 0.0008369

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001590, upper bound: 0.0001487
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001573, upper bound: 0.0001487
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001088, 0.0001103
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002108, 0.0002136
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0017233, 0.0017000
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001518, 0.0001539
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007367, 0.0007468
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001115, 0.0001100
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0002017, 0.0002045
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013335, 0.0013518
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004178, 0.0004235
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008452, 0.0008339

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001526, upper bound: 0.0001532
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001525, upper bound: 0.0001539
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001081, 0.0001109
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002094, 0.0002147
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0017321, 0.0016888
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001508, 0.0001547
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007318, 0.0007506
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001120, 0.0001092
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0002004, 0.0002055
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013247, 0.0013587
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004150, 0.0004257
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008496, 0.0008283

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001508, upper bound: 0.0001543
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001509, upper bound: 0.0001549
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001074, 0.0001112
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002081, 0.0002155
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0017380, 0.0016784
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001499, 0.0001552
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007273, 0.0007531
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001124, 0.0001086
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0001991, 0.0002062
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013166, 0.0013633
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004125, 0.0004271
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008525, 0.0008232

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001454, upper bound: 0.0001586
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001457, upper bound: 0.0001607
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001093, 0.0001094
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002117, 0.0002119
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0017088, 0.0017076
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001525, 0.0001526
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007400, 0.0007405
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001105, 0.0001105
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0002026, 0.0002027
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013395, 0.0013404
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004197, 0.0004199
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008381, 0.0008376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001595, upper bound: 0.0001469
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001579, upper bound: 0.0001465
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001089, 0.0001100
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002110, 0.0002131
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0017191, 0.0017018
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001520, 0.0001535
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007374, 0.0007450
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001112, 0.0001101
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0002019, 0.0002040
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013349, 0.0013485
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004182, 0.0004225
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008432, 0.0008347

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001534, upper bound: 0.0001512
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001534, upper bound: 0.0001512
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001084, 0.0001108
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002099, 0.0002145
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0017304, 0.0016929
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001512, 0.0001546
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007336, 0.0007499
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001119, 0.0001095
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0002009, 0.0002053
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013280, 0.0013574
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004160, 0.0004253
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008487, 0.0008304

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001535, upper bound: 0.0001540
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001531, upper bound: 0.0001547
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001077, 0.0001112
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002086, 0.0002153
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0017367, 0.0016829
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001503, 0.0001551
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007293, 0.0007526
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001123, 0.0001089
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0001997, 0.0002060
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013201, 0.0013623
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004136, 0.0004268
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008518, 0.0008255

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001482, upper bound: 0.0001580
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001484, upper bound: 0.0001606
time: 0.63 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 8.59 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.59
Output dim: 6, lower bound: -0.0001606, upper bound: 0.0001484
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.59
Output dim: 6, lower bound: -0.0001580, upper bound: 0.0001482
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.59
Output dim: 6, lower bound: -0.0001547, upper bound: 0.0001531
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.59
Output dim: 6, lower bound: -0.0001540, upper bound: 0.0001534
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.59
Output dim: 6, lower bound: -0.0001511, upper bound: 0.0001534
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.59
Output dim: 6, lower bound: -0.0001512, upper bound: 0.0001534
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.59
Output dim: 6, lower bound: -0.0001465, upper bound: 0.0001579
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.59
Output dim: 6, lower bound: -0.0001469, upper bound: 0.0001595
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.59
Output dim: 6, lower bound: -0.0001607, upper bound: 0.0001457
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.59
Output dim: 6, lower bound: -0.0001586, upper bound: 0.0001454
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.59
Output dim: 6, lower bound: -0.0001549, upper bound: 0.0001509
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.59
Output dim: 6, lower bound: -0.0001542, upper bound: 0.0001508
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.59
Output dim: 6, lower bound: -0.0001540, upper bound: 0.0001526
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.59
Output dim: 6, lower bound: -0.0001532, upper bound: 0.0001526
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.59
Output dim: 6, lower bound: -0.0001486, upper bound: 0.0001573
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.59
Output dim: 6, lower bound: -0.0001487, upper bound: 0.0001590
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.59
Output dim: 6, lower bound: -0.0001590, upper bound: 0.0001487
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.59
Output dim: 6, lower bound: -0.0001573, upper bound: 0.0001487
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.59
Output dim: 6, lower bound: -0.0001526, upper bound: 0.0001532
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.59
Output dim: 6, lower bound: -0.0001525, upper bound: 0.0001539
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.59
Output dim: 6, lower bound: -0.0001508, upper bound: 0.0001543
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.59
Output dim: 6, lower bound: -0.0001509, upper bound: 0.0001549
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.59
Output dim: 6, lower bound: -0.0001454, upper bound: 0.0001586
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.59
Output dim: 6, lower bound: -0.0001457, upper bound: 0.0001607
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.59
Output dim: 6, lower bound: -0.0001595, upper bound: 0.0001469
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.59
Output dim: 6, lower bound: -0.0001579, upper bound: 0.0001465
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.59
Output dim: 6, lower bound: -0.0001534, upper bound: 0.0001512
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.59
Output dim: 6, lower bound: -0.0001534, upper bound: 0.0001512
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.59
Output dim: 6, lower bound: -0.0001535, upper bound: 0.0001540
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.59
Output dim: 6, lower bound: -0.0001531, upper bound: 0.0001547
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.59
Output dim: 6, lower bound: -0.0001482, upper bound: 0.0001580
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.59
Output dim: 6, lower bound: -0.0001484, upper bound: 0.0001606

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001084, 0.0001058
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002099, 0.0002050
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0016535, 0.0016930
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001512, 0.0001477
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007337, 0.0007165
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001070, 0.0001095
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0002009, 0.0001962
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013280, 0.0012971
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004161, 0.0004064
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008110, 0.0008304

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001341, upper bound: 0.0001243
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001341, upper bound: 0.0001243
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001078, 0.0001060
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002088, 0.0002053
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0016559, 0.0016844
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001504, 0.0001479
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007299, 0.0007176
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001071, 0.0001090
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0001998, 0.0001965
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013213, 0.0012989
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004139, 0.0004069
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008122, 0.0008262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001338, upper bound: 0.0001243
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001338, upper bound: 0.0001243
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001078, 0.0001065
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002089, 0.0002062
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0016635, 0.0016849
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001505, 0.0001486
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007301, 0.0007209
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001076, 0.0001090
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0001999, 0.0001974
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013217, 0.0013049
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004141, 0.0004088
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008159, 0.0008264

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001281, upper bound: 0.0001310
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001281, upper bound: 0.0001309
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001074, 0.0001067
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002080, 0.0002067
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0016675, 0.0016774
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001498, 0.0001489
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007269, 0.0007226
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001079, 0.0001085
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0001990, 0.0001978
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013158, 0.0013080
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004122, 0.0004098
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008179, 0.0008228

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001281, upper bound: 0.0001310
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001281, upper bound: 0.0001309
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001069, 0.0001070
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002070, 0.0002073
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0016723, 0.0016699
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001491, 0.0001494
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007236, 0.0007247
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001082, 0.0001080
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0001981, 0.0001984
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013099, 0.0013118
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004104, 0.0004110
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008203, 0.0008191

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001268, upper bound: 0.0001291
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001268, upper bound: 0.0001291
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001065, 0.0001071
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002063, 0.0002075
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0016740, 0.0016639
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001486, 0.0001495
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007210, 0.0007254
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001083, 0.0001076
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0001974, 0.0001986
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013052, 0.0013131
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004089, 0.0004114
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008211, 0.0008161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001268, upper bound: 0.0001292
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001268, upper bound: 0.0001292
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001063, 0.0001074
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002058, 0.0002081
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0016782, 0.0016602
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001483, 0.0001499
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007194, 0.0007272
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001086, 0.0001074
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0001970, 0.0001991
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013023, 0.0013164
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004080, 0.0004124
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008231, 0.0008143

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001218, upper bound: 0.0001352
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001218, upper bound: 0.0001352
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001059, 0.0001077
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002052, 0.0002086
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0016825, 0.0016550
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001478, 0.0001503
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007172, 0.0007291
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001088, 0.0001071
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0001964, 0.0001996
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0012982, 0.0013198
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004067, 0.0004135
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008252, 0.0008117

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001218, upper bound: 0.0001357
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001218, upper bound: 0.0001357
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001085, 0.0001055
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002102, 0.0002044
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0016490, 0.0016955
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001514, 0.0001473
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007347, 0.0007146
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001067, 0.0001097
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0002012, 0.0001956
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013300, 0.0012935
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004167, 0.0004052
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008088, 0.0008316

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001357, upper bound: 0.0001218
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001357, upper bound: 0.0001218
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001080, 0.0001057
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002092, 0.0002047
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0016515, 0.0016878
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001507, 0.0001475
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007314, 0.0007157
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001068, 0.0001092
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0002002, 0.0001959
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013239, 0.0012955
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004148, 0.0004059
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008100, 0.0008278

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001352, upper bound: 0.0001218
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001352, upper bound: 0.0001218
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001080, 0.0001062
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002092, 0.0002057
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0016594, 0.0016875
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001507, 0.0001482
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007313, 0.0007191
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001073, 0.0001092
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0002002, 0.0001969
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013237, 0.0013016
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004147, 0.0004078
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008139, 0.0008277

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001292, upper bound: 0.0001268
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001292, upper bound: 0.0001268
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001076, 0.0001064
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002084, 0.0002061
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0016628, 0.0016807
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001501, 0.0001485
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007283, 0.0007205
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001076, 0.0001087
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0001994, 0.0001973
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013184, 0.0013043
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004130, 0.0004086
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008156, 0.0008244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001291, upper bound: 0.0001268
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001291, upper bound: 0.0001268
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001073, 0.0001069
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002078, 0.0002071
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0016706, 0.0016759
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001497, 0.0001492
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007263, 0.0007240
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001081, 0.0001084
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0001988, 0.0001982
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013146, 0.0013105
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004119, 0.0004106
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008194, 0.0008220

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001309, upper bound: 0.0001281
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001310, upper bound: 0.0001281
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001069, 0.0001070
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002070, 0.0002073
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0016723, 0.0016698
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001491, 0.0001494
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007236, 0.0007247
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001082, 0.0001080
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0001981, 0.0001984
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013098, 0.0013117
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004104, 0.0004110
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008202, 0.0008190

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001309, upper bound: 0.0001281
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001310, upper bound: 0.0001281
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001066, 0.0001073
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002065, 0.0002079
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0016769, 0.0016659
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001488, 0.0001498
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007219, 0.0007267
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001085, 0.0001078
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0001977, 0.0001990
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013068, 0.0013154
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004094, 0.0004121
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008225, 0.0008171

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001243, upper bound: 0.0001338
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001243, upper bound: 0.0001338
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001062, 0.0001076
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002057, 0.0002084
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0016807, 0.0016593
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001482, 0.0001501
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007191, 0.0007283
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001087, 0.0001073
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0001969, 0.0001994
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013016, 0.0013184
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004078, 0.0004130
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008244, 0.0008139

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001243, upper bound: 0.0001341
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001243, upper bound: 0.0001341
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001076, 0.0001077
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002084, 0.0002086
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0016825, 0.0016807
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001501, 0.0001503
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007283, 0.0007291
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001088, 0.0001087
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0001994, 0.0001996
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013184, 0.0013198
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004130, 0.0004135
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008253, 0.0008244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001341, upper bound: 0.0001243
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001341, upper bound: 0.0001243
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001073, 0.0001078
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002079, 0.0002089
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0016848, 0.0016769
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001498, 0.0001505
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007267, 0.0007301
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001090, 0.0001085
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0001990, 0.0001999
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013154, 0.0013216
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004121, 0.0004141
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008264, 0.0008225

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001338, upper bound: 0.0001243
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001338, upper bound: 0.0001243
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001070, 0.0001083
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002073, 0.0002098
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0016925, 0.0016722
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001494, 0.0001512
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007247, 0.0007334
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001095, 0.0001082
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0001984, 0.0002008
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013117, 0.0013276
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004110, 0.0004159
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008301, 0.0008202

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001281, upper bound: 0.0001310
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001281, upper bound: 0.0001309
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001069, 0.0001086
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002071, 0.0002103
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0016965, 0.0016706
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001492, 0.0001515
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007239, 0.0007352
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001097, 0.0001081
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0001982, 0.0002013
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013105, 0.0013308
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004106, 0.0004169
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008321, 0.0008194

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001281, upper bound: 0.0001310
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001281, upper bound: 0.0001309
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001064, 0.0001089
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002061, 0.0002109
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0017013, 0.0016628
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001485, 0.0001520
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007205, 0.0007372
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001101, 0.0001076
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0001973, 0.0002018
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013043, 0.0013345
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004086, 0.0004181
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008345, 0.0008156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001268, upper bound: 0.0001291
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001268, upper bound: 0.0001291
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001062, 0.0001090
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002057, 0.0002111
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0017029, 0.0016594
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001482, 0.0001521
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007191, 0.0007380
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001102, 0.0001073
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0001969, 0.0002020
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013016, 0.0013358
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004078, 0.0004185
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008353, 0.0008139

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001268, upper bound: 0.0001292
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001268, upper bound: 0.0001292
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001057, 0.0001093
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002047, 0.0002117
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0017072, 0.0016515
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001475, 0.0001525
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007157, 0.0007398
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001104, 0.0001068
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0001959, 0.0002025
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0012955, 0.0013391
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004059, 0.0004195
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008374, 0.0008100

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001218, upper bound: 0.0001352
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001218, upper bound: 0.0001352
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001055, 0.0001095
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002044, 0.0002122
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0017114, 0.0016490
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001473, 0.0001529
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007146, 0.0007416
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001107, 0.0001067
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0001956, 0.0002031
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0012935, 0.0013425
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004052, 0.0004206
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008394, 0.0008088

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001218, upper bound: 0.0001357
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001218, upper bound: 0.0001357
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001077, 0.0001074
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002086, 0.0002080
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0016780, 0.0016825
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001503, 0.0001499
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007291, 0.0007271
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001085, 0.0001088
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0001996, 0.0001991
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013198, 0.0013162
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004135, 0.0004124
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008230, 0.0008252

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001357, upper bound: 0.0001218
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001357, upper bound: 0.0001218
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001074, 0.0001076
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002081, 0.0002083
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0016805, 0.0016782
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001499, 0.0001501
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007272, 0.0007282
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001087, 0.0001086
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0001991, 0.0001994
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013164, 0.0013182
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004124, 0.0004130
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008242, 0.0008231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001352, upper bound: 0.0001218
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001352, upper bound: 0.0001218
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001071, 0.0001081
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002075, 0.0002093
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0016883, 0.0016740
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001495, 0.0001508
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007254, 0.0007316
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001092, 0.0001083
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0001986, 0.0002003
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013131, 0.0013244
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004114, 0.0004149
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008281, 0.0008211

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001291, upper bound: 0.0001268
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001291, upper bound: 0.0001268
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001070, 0.0001083
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002073, 0.0002097
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0016917, 0.0016723
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001494, 0.0001511
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007247, 0.0007331
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001094, 0.0001082
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0001984, 0.0002007
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013118, 0.0013270
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004110, 0.0004158
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008298, 0.0008203

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001291, upper bound: 0.0001268
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001291, upper bound: 0.0001268
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001067, 0.0001088
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002067, 0.0002107
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0016996, 0.0016675
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001489, 0.0001518
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007226, 0.0007365
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001099, 0.0001079
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0001978, 0.0002016
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013080, 0.0013332
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004098, 0.0004177
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008336, 0.0008179

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001309, upper bound: 0.0001281
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001310, upper bound: 0.0001281
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001065, 0.0001089
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002062, 0.0002109
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0017012, 0.0016635
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001486, 0.0001519
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007209, 0.0007372
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001100, 0.0001076
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0001974, 0.0002018
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0013049, 0.0013345
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004088, 0.0004181
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008344, 0.0008159

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001309, upper bound: 0.0001281
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001310, upper bound: 0.0001281
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001060, 0.0001092
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002053, 0.0002115
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0017059, 0.0016559
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001479, 0.0001524
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007176, 0.0007392
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001103, 0.0001071
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0001965, 0.0002024
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0012989, 0.0013381
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004069, 0.0004192
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008367, 0.0008122

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001243, upper bound: 0.0001338
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001243, upper bound: 0.0001338
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060812, 0.0063613, 0.0060812, 0.0063613, -0.0001058, 0.0001094
1: -0.0001392, 0.0004034, -0.0001392, 0.0004034, -0.0002050, 0.0002120
2: 0.0136742, 0.0180504, 0.0136742, 0.0180504, -0.0017097, 0.0016535
3: -0.0041247, -0.0037339, -0.0041247, -0.0037339, -0.0001477, 0.0001527
4: 0.0013237, 0.0032201, 0.0013237, 0.0032201, -0.0007165, 0.0007409
5: -0.0010004, -0.0007173, -0.0010004, -0.0007173, -0.0001106, 0.0001070
6: 0.9915687, 0.9920880, 0.9915687, 0.9920880, -0.0001962, 0.0002028
7: -0.0109867, -0.0075540, -0.0109867, -0.0075540, -0.0012971, 0.0013411
8: -0.0024537, -0.0013783, -0.0024537, -0.0013783, -0.0004064, 0.0004202
9: -0.0045783, -0.0024318, -0.0045783, -0.0024318, -0.0008386, 0.0008110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001243, upper bound: 0.0001341
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001243, upper bound: 0.0001341
time: 0.73 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 8.75 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001341, upper bound: 0.0001243
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001341, upper bound: 0.0001243
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001338, upper bound: 0.0001243
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001338, upper bound: 0.0001243
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001281, upper bound: 0.0001310
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001281, upper bound: 0.0001309
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001281, upper bound: 0.0001310
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001281, upper bound: 0.0001309
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001268, upper bound: 0.0001291
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001268, upper bound: 0.0001291
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001268, upper bound: 0.0001292
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001268, upper bound: 0.0001292
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001218, upper bound: 0.0001352
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001218, upper bound: 0.0001352
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001218, upper bound: 0.0001357
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001218, upper bound: 0.0001357
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001357, upper bound: 0.0001218
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001357, upper bound: 0.0001218
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001352, upper bound: 0.0001218
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001352, upper bound: 0.0001218
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001292, upper bound: 0.0001268
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001292, upper bound: 0.0001268
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001291, upper bound: 0.0001268
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001291, upper bound: 0.0001268
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001309, upper bound: 0.0001281
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001310, upper bound: 0.0001281
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001309, upper bound: 0.0001281
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001310, upper bound: 0.0001281
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001243, upper bound: 0.0001338
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001243, upper bound: 0.0001338
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001243, upper bound: 0.0001341
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001243, upper bound: 0.0001341
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001341, upper bound: 0.0001243
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001341, upper bound: 0.0001243
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001338, upper bound: 0.0001243
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001338, upper bound: 0.0001243
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001281, upper bound: 0.0001310
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001281, upper bound: 0.0001309
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001281, upper bound: 0.0001310
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001281, upper bound: 0.0001309
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001268, upper bound: 0.0001291
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001268, upper bound: 0.0001291
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001268, upper bound: 0.0001292
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001268, upper bound: 0.0001292
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001218, upper bound: 0.0001352
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001218, upper bound: 0.0001352
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001218, upper bound: 0.0001357
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001218, upper bound: 0.0001357
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001357, upper bound: 0.0001218
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001357, upper bound: 0.0001218
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001352, upper bound: 0.0001218
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001352, upper bound: 0.0001218
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001291, upper bound: 0.0001268
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001291, upper bound: 0.0001268
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001291, upper bound: 0.0001268
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001291, upper bound: 0.0001268
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001309, upper bound: 0.0001281
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001310, upper bound: 0.0001281
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001309, upper bound: 0.0001281
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001310, upper bound: 0.0001281
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001243, upper bound: 0.0001338
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001243, upper bound: 0.0001338
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001243, upper bound: 0.0001341
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.75
Output dim: 6, lower bound: -0.0001243, upper bound: 0.0001341

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.99 + 521.36 = 524.35 seconds
