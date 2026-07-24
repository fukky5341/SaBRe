## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00059895


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9906677, 0.9925559, 0.9906677, 0.9925559, -0.0009903, 0.0009903)
1: (-0.0035893, -0.0031189, -0.0035893, -0.0031189, -0.0002468, 0.0002468)
2: (0.0064743, 0.0089675, 0.0064743, 0.0089675, -0.0013077, 0.0013077)
3: (-0.0053547, -0.0042199, -0.0053547, -0.0042199, -0.0005952, 0.0005952)
4: (0.0017810, 0.0022635, 0.0017810, 0.0022635, -0.0002531, 0.0002531)
5: (0.0071024, 0.0102382, 0.0071024, 0.0102382, -0.0016447, 0.0016447)
6: (-0.0010577, -0.0002618, -0.0010577, -0.0002618, -0.0004175, 0.0004175)
7: (-0.0058743, -0.0038151, -0.0058743, -0.0038151, -0.0010801, 0.0010801)
8: (-0.0026534, -0.0015704, -0.0026534, -0.0015704, -0.0005680, 0.0005680)
9: (-0.0000428, 0.0012129, -0.0000428, 0.0012129, -0.0006586, 0.0006586)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.32 + 1.42 = 2.73 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0006668, upper bound: 0.0006668

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006566, upper bound: 0.0006508
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006508, upper bound: 0.0006566
time: 0.58 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.28 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.28
Output dim: 0, lower bound: -0.0006566, upper bound: 0.0006508
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.28
Output dim: 0, lower bound: -0.0006508, upper bound: 0.0006566

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.9906677, 0.9925559, 0.9906677, 0.9925559, -0.0009704, 0.0009687
1: -0.0035893, -0.0031189, -0.0035893, -0.0031189, -0.0002418, 0.0002414
2: 0.0064743, 0.0089675, 0.0064743, 0.0089675, -0.0012791, 0.0012815
3: -0.0053547, -0.0042199, -0.0053547, -0.0042199, -0.0005833, 0.0005822
4: 0.0017810, 0.0022635, 0.0017810, 0.0022635, -0.0002476, 0.0002480
5: 0.0071024, 0.0102382, 0.0071024, 0.0102382, -0.0016088, 0.0016117
6: -0.0010577, -0.0002618, -0.0010577, -0.0002618, -0.0004091, 0.0004083
7: -0.0058743, -0.0038151, -0.0058743, -0.0038151, -0.0010584, 0.0010565
8: -0.0026534, -0.0015704, -0.0026534, -0.0015704, -0.0005566, 0.0005556
9: -0.0000428, 0.0012129, -0.0000428, 0.0012129, -0.0006442, 0.0006454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006377, upper bound: 0.0006349
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006405, upper bound: 0.0006321
time: 0.59 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.9906677, 0.9925559, 0.9906677, 0.9925559, -0.0009687, 0.0009704
1: -0.0035893, -0.0031189, -0.0035893, -0.0031189, -0.0002414, 0.0002418
2: 0.0064743, 0.0089675, 0.0064743, 0.0089675, -0.0012815, 0.0012791
3: -0.0053547, -0.0042199, -0.0053547, -0.0042199, -0.0005822, 0.0005833
4: 0.0017810, 0.0022635, 0.0017810, 0.0022635, -0.0002480, 0.0002476
5: 0.0071024, 0.0102382, 0.0071024, 0.0102382, -0.0016117, 0.0016088
6: -0.0010577, -0.0002618, -0.0010577, -0.0002618, -0.0004083, 0.0004091
7: -0.0058743, -0.0038151, -0.0058743, -0.0038151, -0.0010565, 0.0010584
8: -0.0026534, -0.0015704, -0.0026534, -0.0015704, -0.0005556, 0.0005566
9: -0.0000428, 0.0012129, -0.0000428, 0.0012129, -0.0006454, 0.0006442

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006321, upper bound: 0.0006405
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006349, upper bound: 0.0006378
time: 0.58 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.59 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.59
Output dim: 0, lower bound: -0.0006377, upper bound: 0.0006349
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.59
Output dim: 0, lower bound: -0.0006405, upper bound: 0.0006321
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.59
Output dim: 0, lower bound: -0.0006321, upper bound: 0.0006405
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.59
Output dim: 0, lower bound: -0.0006349, upper bound: 0.0006378

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9906677, 0.9925559, 0.9906677, 0.9925559, -0.0009575, 0.0009568
1: -0.0035893, -0.0031189, -0.0035893, -0.0031189, -0.0002386, 0.0002384
2: 0.0064743, 0.0089675, 0.0064743, 0.0089675, -0.0012634, 0.0012643
3: -0.0053547, -0.0042199, -0.0053547, -0.0042199, -0.0005755, 0.0005750
4: 0.0017810, 0.0022635, 0.0017810, 0.0022635, -0.0002445, 0.0002447
5: 0.0071024, 0.0102382, 0.0071024, 0.0102382, -0.0015890, 0.0015902
6: -0.0010577, -0.0002618, -0.0010577, -0.0002618, -0.0004036, 0.0004033
7: -0.0058743, -0.0038151, -0.0058743, -0.0038151, -0.0010443, 0.0010435
8: -0.0026534, -0.0015704, -0.0026534, -0.0015704, -0.0005492, 0.0005488
9: -0.0000428, 0.0012129, -0.0000428, 0.0012129, -0.0006363, 0.0006368

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006212, upper bound: 0.0006158
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006214, upper bound: 0.0006128
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9906677, 0.9925559, 0.9906677, 0.9925559, -0.0009704, 0.0009557
1: -0.0035893, -0.0031189, -0.0035893, -0.0031189, -0.0002418, 0.0002381
2: 0.0064743, 0.0089675, 0.0064743, 0.0089675, -0.0012620, 0.0012815
3: -0.0053547, -0.0042199, -0.0053547, -0.0042199, -0.0005833, 0.0005744
4: 0.0017810, 0.0022635, 0.0017810, 0.0022635, -0.0002443, 0.0002480
5: 0.0071024, 0.0102382, 0.0071024, 0.0102382, -0.0015873, 0.0016117
6: -0.0010577, -0.0002618, -0.0010577, -0.0002618, -0.0004091, 0.0004029
7: -0.0058743, -0.0038151, -0.0058743, -0.0038151, -0.0010584, 0.0010423
8: -0.0026534, -0.0015704, -0.0026534, -0.0015704, -0.0005566, 0.0005482
9: -0.0000428, 0.0012129, -0.0000428, 0.0012129, -0.0006356, 0.0006454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006221, upper bound: 0.0006147
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006232, upper bound: 0.0006125
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9906677, 0.9925559, 0.9906677, 0.9925559, -0.0009557, 0.0009586
1: -0.0035893, -0.0031189, -0.0035893, -0.0031189, -0.0002381, 0.0002389
2: 0.0064743, 0.0089675, 0.0064743, 0.0089675, -0.0012658, 0.0012620
3: -0.0053547, -0.0042199, -0.0053547, -0.0042199, -0.0005744, 0.0005761
4: 0.0017810, 0.0022635, 0.0017810, 0.0022635, -0.0002450, 0.0002443
5: 0.0071024, 0.0102382, 0.0071024, 0.0102382, -0.0015920, 0.0015873
6: -0.0010577, -0.0002618, -0.0010577, -0.0002618, -0.0004029, 0.0004041
7: -0.0058743, -0.0038151, -0.0058743, -0.0038151, -0.0010423, 0.0010455
8: -0.0026534, -0.0015704, -0.0026534, -0.0015704, -0.0005482, 0.0005498
9: -0.0000428, 0.0012129, -0.0000428, 0.0012129, -0.0006375, 0.0006356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006125, upper bound: 0.0006232
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006146, upper bound: 0.0006221
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9906677, 0.9925559, 0.9906677, 0.9925559, -0.0009687, 0.0009575
1: -0.0035893, -0.0031189, -0.0035893, -0.0031189, -0.0002414, 0.0002386
2: 0.0064743, 0.0089675, 0.0064743, 0.0089675, -0.0012643, 0.0012791
3: -0.0053547, -0.0042199, -0.0053547, -0.0042199, -0.0005822, 0.0005755
4: 0.0017810, 0.0022635, 0.0017810, 0.0022635, -0.0002447, 0.0002476
5: 0.0071024, 0.0102382, 0.0071024, 0.0102382, -0.0015902, 0.0016088
6: -0.0010577, -0.0002618, -0.0010577, -0.0002618, -0.0004083, 0.0004036
7: -0.0058743, -0.0038151, -0.0058743, -0.0038151, -0.0010565, 0.0010443
8: -0.0026534, -0.0015704, -0.0026534, -0.0015704, -0.0005556, 0.0005492
9: -0.0000428, 0.0012129, -0.0000428, 0.0012129, -0.0006368, 0.0006442

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006127, upper bound: 0.0006214
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006158, upper bound: 0.0006212
time: 0.64 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.61 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -0.0006212, upper bound: 0.0006158
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -0.0006214, upper bound: 0.0006128
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -0.0006221, upper bound: 0.0006147
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -0.0006232, upper bound: 0.0006125
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -0.0006125, upper bound: 0.0006232
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -0.0006146, upper bound: 0.0006221
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -0.0006127, upper bound: 0.0006214
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -0.0006158, upper bound: 0.0006212

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9906677, 0.9925559, 0.9906677, 0.9925559, -0.0009314, 0.0009325
1: -0.0035893, -0.0031189, -0.0035893, -0.0031189, -0.0002321, 0.0002323
2: 0.0064743, 0.0089675, 0.0064743, 0.0089675, -0.0012313, 0.0012300
3: -0.0053547, -0.0042199, -0.0053547, -0.0042199, -0.0005598, 0.0005604
4: 0.0017810, 0.0022635, 0.0017810, 0.0022635, -0.0002383, 0.0002381
5: 0.0071024, 0.0102382, 0.0071024, 0.0102382, -0.0015486, 0.0015470
6: -0.0010577, -0.0002618, -0.0010577, -0.0002618, -0.0003926, 0.0003931
7: -0.0058743, -0.0038151, -0.0058743, -0.0038151, -0.0010159, 0.0010170
8: -0.0026534, -0.0015704, -0.0026534, -0.0015704, -0.0005342, 0.0005348
9: -0.0000428, 0.0012129, -0.0000428, 0.0012129, -0.0006201, 0.0006195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005857, upper bound: 0.0006053
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006095, upper bound: 0.0005734
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9906677, 0.9925559, 0.9906677, 0.9925559, -0.0009332, 0.0009303
1: -0.0035893, -0.0031189, -0.0035893, -0.0031189, -0.0002325, 0.0002318
2: 0.0064743, 0.0089675, 0.0064743, 0.0089675, -0.0012284, 0.0012322
3: -0.0053547, -0.0042199, -0.0053547, -0.0042199, -0.0005609, 0.0005591
4: 0.0017810, 0.0022635, 0.0017810, 0.0022635, -0.0002378, 0.0002385
5: 0.0071024, 0.0102382, 0.0071024, 0.0102382, -0.0015450, 0.0015498
6: -0.0010577, -0.0002618, -0.0010577, -0.0002618, -0.0003934, 0.0003921
7: -0.0058743, -0.0038151, -0.0058743, -0.0038151, -0.0010177, 0.0010146
8: -0.0026534, -0.0015704, -0.0026534, -0.0015704, -0.0005352, 0.0005336
9: -0.0000428, 0.0012129, -0.0000428, 0.0012129, -0.0006187, 0.0006206

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005825, upper bound: 0.0006020
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006097, upper bound: 0.0005734
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9906677, 0.9925559, 0.9906677, 0.9925559, -0.0009449, 0.0009314
1: -0.0035893, -0.0031189, -0.0035893, -0.0031189, -0.0002355, 0.0002321
2: 0.0064743, 0.0089675, 0.0064743, 0.0089675, -0.0012299, 0.0012478
3: -0.0053547, -0.0042199, -0.0053547, -0.0042199, -0.0005679, 0.0005598
4: 0.0017810, 0.0022635, 0.0017810, 0.0022635, -0.0002380, 0.0002415
5: 0.0071024, 0.0102382, 0.0071024, 0.0102382, -0.0015469, 0.0015694
6: -0.0010577, -0.0002618, -0.0010577, -0.0002618, -0.0003983, 0.0003926
7: -0.0058743, -0.0038151, -0.0058743, -0.0038151, -0.0010306, 0.0010158
8: -0.0026534, -0.0015704, -0.0026534, -0.0015704, -0.0005420, 0.0005342
9: -0.0000428, 0.0012129, -0.0000428, 0.0012129, -0.0006194, 0.0006285

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005866, upper bound: 0.0006039
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006105, upper bound: 0.0005722
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9906677, 0.9925559, 0.9906677, 0.9925559, -0.0009467, 0.0009293
1: -0.0035893, -0.0031189, -0.0035893, -0.0031189, -0.0002359, 0.0002315
2: 0.0064743, 0.0089675, 0.0064743, 0.0089675, -0.0012271, 0.0012501
3: -0.0053547, -0.0042199, -0.0053547, -0.0042199, -0.0005690, 0.0005585
4: 0.0017810, 0.0022635, 0.0017810, 0.0022635, -0.0002375, 0.0002419
5: 0.0071024, 0.0102382, 0.0071024, 0.0102382, -0.0015433, 0.0015722
6: -0.0010577, -0.0002618, -0.0010577, -0.0002618, -0.0003991, 0.0003917
7: -0.0058743, -0.0038151, -0.0058743, -0.0038151, -0.0010325, 0.0010135
8: -0.0026534, -0.0015704, -0.0026534, -0.0015704, -0.0005430, 0.0005330
9: -0.0000428, 0.0012129, -0.0000428, 0.0012129, -0.0006180, 0.0006296

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005851, upper bound: 0.0006016
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006117, upper bound: 0.0005727
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9906677, 0.9925559, 0.9906677, 0.9925559, -0.0009293, 0.0009343
1: -0.0035893, -0.0031189, -0.0035893, -0.0031189, -0.0002315, 0.0002328
2: 0.0064743, 0.0089675, 0.0064743, 0.0089675, -0.0012337, 0.0012271
3: -0.0053547, -0.0042199, -0.0053547, -0.0042199, -0.0005585, 0.0005615
4: 0.0017810, 0.0022635, 0.0017810, 0.0022635, -0.0002388, 0.0002375
5: 0.0071024, 0.0102382, 0.0071024, 0.0102382, -0.0015517, 0.0015433
6: -0.0010577, -0.0002618, -0.0010577, -0.0002618, -0.0003917, 0.0003938
7: -0.0058743, -0.0038151, -0.0058743, -0.0038151, -0.0010135, 0.0010190
8: -0.0026534, -0.0015704, -0.0026534, -0.0015704, -0.0005330, 0.0005359
9: -0.0000428, 0.0012129, -0.0000428, 0.0012129, -0.0006214, 0.0006180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005727, upper bound: 0.0006117
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006016, upper bound: 0.0005851
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9906677, 0.9925559, 0.9906677, 0.9925559, -0.0009314, 0.0009325
1: -0.0035893, -0.0031189, -0.0035893, -0.0031189, -0.0002321, 0.0002324
2: 0.0064743, 0.0089675, 0.0064743, 0.0089675, -0.0012314, 0.0012299
3: -0.0053547, -0.0042199, -0.0053547, -0.0042199, -0.0005598, 0.0005605
4: 0.0017810, 0.0022635, 0.0017810, 0.0022635, -0.0002383, 0.0002380
5: 0.0071024, 0.0102382, 0.0071024, 0.0102382, -0.0015488, 0.0015469
6: -0.0010577, -0.0002618, -0.0010577, -0.0002618, -0.0003926, 0.0003931
7: -0.0058743, -0.0038151, -0.0058743, -0.0038151, -0.0010158, 0.0010171
8: -0.0026534, -0.0015704, -0.0026534, -0.0015704, -0.0005342, 0.0005349
9: -0.0000428, 0.0012129, -0.0000428, 0.0012129, -0.0006202, 0.0006194

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005722, upper bound: 0.0006105
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006038, upper bound: 0.0005866
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9906677, 0.9925559, 0.9906677, 0.9925559, -0.0009428, 0.0009332
1: -0.0035893, -0.0031189, -0.0035893, -0.0031189, -0.0002349, 0.0002325
2: 0.0064743, 0.0089675, 0.0064743, 0.0089675, -0.0012322, 0.0012449
3: -0.0053547, -0.0042199, -0.0053547, -0.0042199, -0.0005666, 0.0005609
4: 0.0017810, 0.0022635, 0.0017810, 0.0022635, -0.0002385, 0.0002409
5: 0.0071024, 0.0102382, 0.0071024, 0.0102382, -0.0015498, 0.0015657
6: -0.0010577, -0.0002618, -0.0010577, -0.0002618, -0.0003974, 0.0003934
7: -0.0058743, -0.0038151, -0.0058743, -0.0038151, -0.0010282, 0.0010177
8: -0.0026534, -0.0015704, -0.0026534, -0.0015704, -0.0005407, 0.0005352
9: -0.0000428, 0.0012129, -0.0000428, 0.0012129, -0.0006206, 0.0006270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005734, upper bound: 0.0006097
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006020, upper bound: 0.0005825
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9906677, 0.9925559, 0.9906677, 0.9925559, -0.0009449, 0.0009314
1: -0.0035893, -0.0031189, -0.0035893, -0.0031189, -0.0002354, 0.0002321
2: 0.0064743, 0.0089675, 0.0064743, 0.0089675, -0.0012300, 0.0012477
3: -0.0053547, -0.0042199, -0.0053547, -0.0042199, -0.0005679, 0.0005598
4: 0.0017810, 0.0022635, 0.0017810, 0.0022635, -0.0002381, 0.0002415
5: 0.0071024, 0.0102382, 0.0071024, 0.0102382, -0.0015470, 0.0015693
6: -0.0010577, -0.0002618, -0.0010577, -0.0002618, -0.0003983, 0.0003926
7: -0.0058743, -0.0038151, -0.0058743, -0.0038151, -0.0010305, 0.0010159
8: -0.0026534, -0.0015704, -0.0026534, -0.0015704, -0.0005420, 0.0005342
9: -0.0000428, 0.0012129, -0.0000428, 0.0012129, -0.0006195, 0.0006284

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005734, upper bound: 0.0006095
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006053, upper bound: 0.0005857
time: 0.61 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.61 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0005857, upper bound: 0.0006053
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0006095, upper bound: 0.0005734
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0005825, upper bound: 0.0006020
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0006097, upper bound: 0.0005734
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0005866, upper bound: 0.0006039
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0006105, upper bound: 0.0005722
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0005851, upper bound: 0.0006016
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0006117, upper bound: 0.0005727
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0005727, upper bound: 0.0006117
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0006016, upper bound: 0.0005851
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0005722, upper bound: 0.0006105
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0006038, upper bound: 0.0005866
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0005734, upper bound: 0.0006097
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0006020, upper bound: 0.0005825
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0005734, upper bound: 0.0006095
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0006053, upper bound: 0.0005857

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9906677, 0.9925559, 0.9906677, 0.9925559, -0.0008958, 0.0009055
1: -0.0035893, -0.0031189, -0.0035893, -0.0031189, -0.0002232, 0.0002256
2: 0.0064743, 0.0089675, 0.0064743, 0.0089675, -0.0011957, 0.0011828
3: -0.0053547, -0.0042199, -0.0053547, -0.0042199, -0.0005384, 0.0005442
4: 0.0017810, 0.0022635, 0.0017810, 0.0022635, -0.0002314, 0.0002289
5: 0.0071024, 0.0102382, 0.0071024, 0.0102382, -0.0015039, 0.0014877
6: -0.0010577, -0.0002618, -0.0010577, -0.0002618, -0.0003776, 0.0003817
7: -0.0058743, -0.0038151, -0.0058743, -0.0038151, -0.0009770, 0.0009876
8: -0.0026534, -0.0015704, -0.0026534, -0.0015704, -0.0005138, 0.0005193
9: -0.0000428, 0.0012129, -0.0000428, 0.0012129, -0.0006022, 0.0005957

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005677, upper bound: 0.0005824
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005653, upper bound: 0.0005881
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9906677, 0.9925559, 0.9906677, 0.9925559, -0.0009033, 0.0008968
1: -0.0035893, -0.0031189, -0.0035893, -0.0031189, -0.0002251, 0.0002235
2: 0.0064743, 0.0089675, 0.0064743, 0.0089675, -0.0011842, 0.0011929
3: -0.0053547, -0.0042199, -0.0053547, -0.0042199, -0.0005429, 0.0005390
4: 0.0017810, 0.0022635, 0.0017810, 0.0022635, -0.0002292, 0.0002309
5: 0.0071024, 0.0102382, 0.0071024, 0.0102382, -0.0014894, 0.0015003
6: -0.0010577, -0.0002618, -0.0010577, -0.0002618, -0.0003808, 0.0003780
7: -0.0058743, -0.0038151, -0.0058743, -0.0038151, -0.0009852, 0.0009781
8: -0.0026534, -0.0015704, -0.0026534, -0.0015704, -0.0005181, 0.0005143
9: -0.0000428, 0.0012129, -0.0000428, 0.0012129, -0.0005964, 0.0006008

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005907, upper bound: 0.0005558
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005876, upper bound: 0.0005566
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9906677, 0.9925559, 0.9906677, 0.9925559, -0.0008975, 0.0009029
1: -0.0035893, -0.0031189, -0.0035893, -0.0031189, -0.0002236, 0.0002250
2: 0.0064743, 0.0089675, 0.0064743, 0.0089675, -0.0011923, 0.0011851
3: -0.0053547, -0.0042199, -0.0053547, -0.0042199, -0.0005394, 0.0005427
4: 0.0017810, 0.0022635, 0.0017810, 0.0022635, -0.0002308, 0.0002294
5: 0.0071024, 0.0102382, 0.0071024, 0.0102382, -0.0014996, 0.0014905
6: -0.0010577, -0.0002618, -0.0010577, -0.0002618, -0.0003783, 0.0003806
7: -0.0058743, -0.0038151, -0.0058743, -0.0038151, -0.0009788, 0.0009848
8: -0.0026534, -0.0015704, -0.0026534, -0.0015704, -0.0005148, 0.0005179
9: -0.0000428, 0.0012129, -0.0000428, 0.0012129, -0.0006005, 0.0005969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005649, upper bound: 0.0005798
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005622, upper bound: 0.0005834
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9906677, 0.9925559, 0.9906677, 0.9925559, -0.0009049, 0.0008946
1: -0.0035893, -0.0031189, -0.0035893, -0.0031189, -0.0002255, 0.0002229
2: 0.0064743, 0.0089675, 0.0064743, 0.0089675, -0.0011813, 0.0011950
3: -0.0053547, -0.0042199, -0.0053547, -0.0042199, -0.0005439, 0.0005377
4: 0.0017810, 0.0022635, 0.0017810, 0.0022635, -0.0002286, 0.0002313
5: 0.0071024, 0.0102382, 0.0071024, 0.0102382, -0.0014857, 0.0015030
6: -0.0010577, -0.0002618, -0.0010577, -0.0002618, -0.0003815, 0.0003771
7: -0.0058743, -0.0038151, -0.0058743, -0.0038151, -0.0009870, 0.0009757
8: -0.0026534, -0.0015704, -0.0026534, -0.0015704, -0.0005190, 0.0005131
9: -0.0000428, 0.0012129, -0.0000428, 0.0012129, -0.0005949, 0.0006019

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005917, upper bound: 0.0005559
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005878, upper bound: 0.0005566
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9906677, 0.9925559, 0.9906677, 0.9925559, -0.0009080, 0.0009038
1: -0.0035893, -0.0031189, -0.0035893, -0.0031189, -0.0002262, 0.0002252
2: 0.0064743, 0.0089675, 0.0064743, 0.0089675, -0.0011935, 0.0011989
3: -0.0053547, -0.0042199, -0.0053547, -0.0042199, -0.0005457, 0.0005432
4: 0.0017810, 0.0022635, 0.0017810, 0.0022635, -0.0002310, 0.0002321
5: 0.0071024, 0.0102382, 0.0071024, 0.0102382, -0.0015011, 0.0015080
6: -0.0010577, -0.0002618, -0.0010577, -0.0002618, -0.0003827, 0.0003810
7: -0.0058743, -0.0038151, -0.0058743, -0.0038151, -0.0009903, 0.0009858
8: -0.0026534, -0.0015704, -0.0026534, -0.0015704, -0.0005208, 0.0005184
9: -0.0000428, 0.0012129, -0.0000428, 0.0012129, -0.0006011, 0.0006038

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005694, upper bound: 0.0005814
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005674, upper bound: 0.0005857
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9906677, 0.9925559, 0.9906677, 0.9925559, -0.0009155, 0.0008957
1: -0.0035893, -0.0031189, -0.0035893, -0.0031189, -0.0002281, 0.0002232
2: 0.0064743, 0.0089675, 0.0064743, 0.0089675, -0.0011828, 0.0012090
3: -0.0053547, -0.0042199, -0.0053547, -0.0042199, -0.0005503, 0.0005383
4: 0.0017810, 0.0022635, 0.0017810, 0.0022635, -0.0002289, 0.0002340
5: 0.0071024, 0.0102382, 0.0071024, 0.0102382, -0.0014876, 0.0015205
6: -0.0010577, -0.0002618, -0.0010577, -0.0002618, -0.0003859, 0.0003776
7: -0.0058743, -0.0038151, -0.0058743, -0.0038151, -0.0009985, 0.0009769
8: -0.0026534, -0.0015704, -0.0026534, -0.0015704, -0.0005251, 0.0005137
9: -0.0000428, 0.0012129, -0.0000428, 0.0012129, -0.0005957, 0.0006089

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005926, upper bound: 0.0005528
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005898, upper bound: 0.0005548
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9906677, 0.9925559, 0.9906677, 0.9925559, -0.0009097, 0.0009019
1: -0.0035893, -0.0031189, -0.0035893, -0.0031189, -0.0002267, 0.0002247
2: 0.0064743, 0.0089675, 0.0064743, 0.0089675, -0.0011910, 0.0012012
3: -0.0053547, -0.0042199, -0.0053547, -0.0042199, -0.0005467, 0.0005421
4: 0.0017810, 0.0022635, 0.0017810, 0.0022635, -0.0002305, 0.0002325
5: 0.0071024, 0.0102382, 0.0071024, 0.0102382, -0.0014980, 0.0015108
6: -0.0010577, -0.0002618, -0.0010577, -0.0002618, -0.0003835, 0.0003802
7: -0.0058743, -0.0038151, -0.0058743, -0.0038151, -0.0009921, 0.0009837
8: -0.0026534, -0.0015704, -0.0026534, -0.0015704, -0.0005217, 0.0005173
9: -0.0000428, 0.0012129, -0.0000428, 0.0012129, -0.0005998, 0.0006050

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005684, upper bound: 0.0005792
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005662, upper bound: 0.0005827
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9906677, 0.9925559, 0.9906677, 0.9925559, -0.0009171, 0.0008936
1: -0.0035893, -0.0031189, -0.0035893, -0.0031189, -0.0002285, 0.0002227
2: 0.0064743, 0.0089675, 0.0064743, 0.0089675, -0.0011799, 0.0012111
3: -0.0053547, -0.0042199, -0.0053547, -0.0042199, -0.0005512, 0.0005371
4: 0.0017810, 0.0022635, 0.0017810, 0.0022635, -0.0002284, 0.0002344
5: 0.0071024, 0.0102382, 0.0071024, 0.0102382, -0.0014841, 0.0015232
6: -0.0010577, -0.0002618, -0.0010577, -0.0002618, -0.0003866, 0.0003767
7: -0.0058743, -0.0038151, -0.0058743, -0.0038151, -0.0010003, 0.0009746
8: -0.0026534, -0.0015704, -0.0026534, -0.0015704, -0.0005260, 0.0005125
9: -0.0000428, 0.0012129, -0.0000428, 0.0012129, -0.0005943, 0.0006100

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005947, upper bound: 0.0005533
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005906, upper bound: 0.0005552
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9906677, 0.9925559, 0.9906677, 0.9925559, -0.0008936, 0.0009066
1: -0.0035893, -0.0031189, -0.0035893, -0.0031189, -0.0002227, 0.0002259
2: 0.0064743, 0.0089675, 0.0064743, 0.0089675, -0.0011971, 0.0011799
3: -0.0053547, -0.0042199, -0.0053547, -0.0042199, -0.0005371, 0.0005449
4: 0.0017810, 0.0022635, 0.0017810, 0.0022635, -0.0002317, 0.0002284
5: 0.0071024, 0.0102382, 0.0071024, 0.0102382, -0.0015057, 0.0014841
6: -0.0010577, -0.0002618, -0.0010577, -0.0002618, -0.0003767, 0.0003822
7: -0.0058743, -0.0038151, -0.0058743, -0.0038151, -0.0009746, 0.0009888
8: -0.0026534, -0.0015704, -0.0026534, -0.0015704, -0.0005125, 0.0005200
9: -0.0000428, 0.0012129, -0.0000428, 0.0012129, -0.0006029, 0.0005943

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005552, upper bound: 0.0005906
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005533, upper bound: 0.0005947
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9906677, 0.9925559, 0.9906677, 0.9925559, -0.0009019, 0.0008986
1: -0.0035893, -0.0031189, -0.0035893, -0.0031189, -0.0002247, 0.0002239
2: 0.0064743, 0.0089675, 0.0064743, 0.0089675, -0.0011866, 0.0011910
3: -0.0053547, -0.0042199, -0.0053547, -0.0042199, -0.0005421, 0.0005401
4: 0.0017810, 0.0022635, 0.0017810, 0.0022635, -0.0002297, 0.0002305
5: 0.0071024, 0.0102382, 0.0071024, 0.0102382, -0.0014924, 0.0014980
6: -0.0010577, -0.0002618, -0.0010577, -0.0002618, -0.0003802, 0.0003788
7: -0.0058743, -0.0038151, -0.0058743, -0.0038151, -0.0009837, 0.0009800
8: -0.0026534, -0.0015704, -0.0026534, -0.0015704, -0.0005173, 0.0005154
9: -0.0000428, 0.0012129, -0.0000428, 0.0012129, -0.0005976, 0.0005998

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005827, upper bound: 0.0005662
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005793, upper bound: 0.0005684
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9906677, 0.9925559, 0.9906677, 0.9925559, -0.0008957, 0.0009046
1: -0.0035893, -0.0031189, -0.0035893, -0.0031189, -0.0002232, 0.0002254
2: 0.0064743, 0.0089675, 0.0064743, 0.0089675, -0.0011945, 0.0011828
3: -0.0053547, -0.0042199, -0.0053547, -0.0042199, -0.0005383, 0.0005437
4: 0.0017810, 0.0022635, 0.0017810, 0.0022635, -0.0002312, 0.0002289
5: 0.0071024, 0.0102382, 0.0071024, 0.0102382, -0.0015023, 0.0014876
6: -0.0010577, -0.0002618, -0.0010577, -0.0002618, -0.0003776, 0.0003813
7: -0.0058743, -0.0038151, -0.0058743, -0.0038151, -0.0009769, 0.0009866
8: -0.0026534, -0.0015704, -0.0026534, -0.0015704, -0.0005137, 0.0005188
9: -0.0000428, 0.0012129, -0.0000428, 0.0012129, -0.0006016, 0.0005957

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005548, upper bound: 0.0005898
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005528, upper bound: 0.0005926
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9906677, 0.9925559, 0.9906677, 0.9925559, -0.0009038, 0.0008968
1: -0.0035893, -0.0031189, -0.0035893, -0.0031189, -0.0002252, 0.0002235
2: 0.0064743, 0.0089675, 0.0064743, 0.0089675, -0.0011843, 0.0011935
3: -0.0053547, -0.0042199, -0.0053547, -0.0042199, -0.0005432, 0.0005390
4: 0.0017810, 0.0022635, 0.0017810, 0.0022635, -0.0002292, 0.0002310
5: 0.0071024, 0.0102382, 0.0071024, 0.0102382, -0.0014895, 0.0015011
6: -0.0010577, -0.0002618, -0.0010577, -0.0002618, -0.0003810, 0.0003781
7: -0.0058743, -0.0038151, -0.0058743, -0.0038151, -0.0009858, 0.0009781
8: -0.0026534, -0.0015704, -0.0026534, -0.0015704, -0.0005184, 0.0005144
9: -0.0000428, 0.0012129, -0.0000428, 0.0012129, -0.0005965, 0.0006011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005858, upper bound: 0.0005674
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005814, upper bound: 0.0005694
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9906677, 0.9925559, 0.9906677, 0.9925559, -0.0009058, 0.0009049
1: -0.0035893, -0.0031189, -0.0035893, -0.0031189, -0.0002257, 0.0002255
2: 0.0064743, 0.0089675, 0.0064743, 0.0089675, -0.0011950, 0.0011960
3: -0.0053547, -0.0042199, -0.0053547, -0.0042199, -0.0005444, 0.0005439
4: 0.0017810, 0.0022635, 0.0017810, 0.0022635, -0.0002313, 0.0002315
5: 0.0071024, 0.0102382, 0.0071024, 0.0102382, -0.0015030, 0.0015043
6: -0.0010577, -0.0002618, -0.0010577, -0.0002618, -0.0003818, 0.0003815
7: -0.0058743, -0.0038151, -0.0058743, -0.0038151, -0.0009879, 0.0009870
8: -0.0026534, -0.0015704, -0.0026534, -0.0015704, -0.0005195, 0.0005190
9: -0.0000428, 0.0012129, -0.0000428, 0.0012129, -0.0006019, 0.0006024

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005565, upper bound: 0.0005878
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005559, upper bound: 0.0005917
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9906677, 0.9925559, 0.9906677, 0.9925559, -0.0009141, 0.0008975
1: -0.0035893, -0.0031189, -0.0035893, -0.0031189, -0.0002278, 0.0002236
2: 0.0064743, 0.0089675, 0.0064743, 0.0089675, -0.0011851, 0.0012071
3: -0.0053547, -0.0042199, -0.0053547, -0.0042199, -0.0005494, 0.0005394
4: 0.0017810, 0.0022635, 0.0017810, 0.0022635, -0.0002294, 0.0002336
5: 0.0071024, 0.0102382, 0.0071024, 0.0102382, -0.0014905, 0.0015182
6: -0.0010577, -0.0002618, -0.0010577, -0.0002618, -0.0003853, 0.0003783
7: -0.0058743, -0.0038151, -0.0058743, -0.0038151, -0.0009970, 0.0009788
8: -0.0026534, -0.0015704, -0.0026534, -0.0015704, -0.0005243, 0.0005148
9: -0.0000428, 0.0012129, -0.0000428, 0.0012129, -0.0005969, 0.0006080

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005834, upper bound: 0.0005623
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005798, upper bound: 0.0005649
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9906677, 0.9925559, 0.9906677, 0.9925559, -0.0009079, 0.0009033
1: -0.0035893, -0.0031189, -0.0035893, -0.0031189, -0.0002262, 0.0002251
2: 0.0064743, 0.0089675, 0.0064743, 0.0089675, -0.0011929, 0.0011989
3: -0.0053547, -0.0042199, -0.0053547, -0.0042199, -0.0005457, 0.0005429
4: 0.0017810, 0.0022635, 0.0017810, 0.0022635, -0.0002309, 0.0002320
5: 0.0071024, 0.0102382, 0.0071024, 0.0102382, -0.0015003, 0.0015079
6: -0.0010577, -0.0002618, -0.0010577, -0.0002618, -0.0003827, 0.0003808
7: -0.0058743, -0.0038151, -0.0058743, -0.0038151, -0.0009902, 0.0009852
8: -0.0026534, -0.0015704, -0.0026534, -0.0015704, -0.0005207, 0.0005181
9: -0.0000428, 0.0012129, -0.0000428, 0.0012129, -0.0006008, 0.0006038

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005566, upper bound: 0.0005875
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005558, upper bound: 0.0005907
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9906677, 0.9925559, 0.9906677, 0.9925559, -0.0009160, 0.0008958
1: -0.0035893, -0.0031189, -0.0035893, -0.0031189, -0.0002282, 0.0002232
2: 0.0064743, 0.0089675, 0.0064743, 0.0089675, -0.0011828, 0.0012096
3: -0.0053547, -0.0042199, -0.0053547, -0.0042199, -0.0005506, 0.0005384
4: 0.0017810, 0.0022635, 0.0017810, 0.0022635, -0.0002289, 0.0002341
5: 0.0071024, 0.0102382, 0.0071024, 0.0102382, -0.0014877, 0.0015214
6: -0.0010577, -0.0002618, -0.0010577, -0.0002618, -0.0003861, 0.0003776
7: -0.0058743, -0.0038151, -0.0058743, -0.0038151, -0.0009991, 0.0009770
8: -0.0026534, -0.0015704, -0.0026534, -0.0015704, -0.0005254, 0.0005138
9: -0.0000428, 0.0012129, -0.0000428, 0.0012129, -0.0005957, 0.0006092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005881, upper bound: 0.0005653
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005824, upper bound: 0.0005677
time: 0.66 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.66 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0005677, upper bound: 0.0005824
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0005653, upper bound: 0.0005881
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0005907, upper bound: 0.0005558
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0005876, upper bound: 0.0005566
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0005649, upper bound: 0.0005798
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0005622, upper bound: 0.0005834
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0005917, upper bound: 0.0005559
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0005878, upper bound: 0.0005566
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0005694, upper bound: 0.0005814
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0005674, upper bound: 0.0005857
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0005926, upper bound: 0.0005528
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0005898, upper bound: 0.0005548
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0005684, upper bound: 0.0005792
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0005662, upper bound: 0.0005827
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0005947, upper bound: 0.0005533
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0005906, upper bound: 0.0005552
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0005552, upper bound: 0.0005906
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0005533, upper bound: 0.0005947
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0005827, upper bound: 0.0005662
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0005793, upper bound: 0.0005684
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0005548, upper bound: 0.0005898
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0005528, upper bound: 0.0005926
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0005858, upper bound: 0.0005674
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0005814, upper bound: 0.0005694
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0005565, upper bound: 0.0005878
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0005559, upper bound: 0.0005917
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0005834, upper bound: 0.0005623
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0005798, upper bound: 0.0005649
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0005566, upper bound: 0.0005875
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0005558, upper bound: 0.0005907
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0005881, upper bound: 0.0005653
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0005824, upper bound: 0.0005677

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.73 + 79.49 = 82.22 seconds
