## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 8.477e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006868, 0.0006868)
1: (-0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000929, 0.0000929)
2: (0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0008402, 0.0008402)
3: (1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002329, 0.0002329)
4: (-0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001238, 0.0001238)
5: (0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0005224, 0.0005224)
6: (-0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000420, 0.0000420)
7: (-0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0014098, 0.0014098)
8: (-0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0012501, 0.0012501)
9: (-0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0005667, 0.0005667)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.42 + 1.35 = 2.77 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0001204, upper bound: 0.0001204

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001200, upper bound: 0.0001164
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001163, upper bound: 0.0001201
time: 0.54 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.22 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.22
Output dim: 3, lower bound: -0.0001200, upper bound: 0.0001164
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.22
Output dim: 3, lower bound: -0.0001163, upper bound: 0.0001201

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006769, 0.0006768
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000911, 0.0000913
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0008278, 0.0008277
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002318, 0.0002316
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001218, 0.0001218
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0005148, 0.0005147
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000417, 0.0000417
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0013924, 0.0013933
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0012301, 0.0012295
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0005575, 0.0005578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001146, upper bound: 0.0001087
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001124, upper bound: 0.0001111
time: 0.55 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006768, 0.0006769
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000913, 0.0000911
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0008277, 0.0008278
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002316, 0.0002318
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001218, 0.0001218
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0005147, 0.0005148
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000417, 0.0000417
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0013933, 0.0013924
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0012295, 0.0012301
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0005578, 0.0005575

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001111, upper bound: 0.0001125
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001087, upper bound: 0.0001146
time: 0.54 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.52 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.52
Output dim: 3, lower bound: -0.0001146, upper bound: 0.0001087
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.52
Output dim: 3, lower bound: -0.0001124, upper bound: 0.0001111
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.52
Output dim: 3, lower bound: -0.0001111, upper bound: 0.0001125
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.52
Output dim: 3, lower bound: -0.0001087, upper bound: 0.0001146

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006571, 0.0006595
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000908, 0.0000910
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0008048, 0.0008086
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002245, 0.0002227
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001195, 0.0001189
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004998, 0.0005018
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000399, 0.0000397
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0013434, 0.0013427
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0012096, 0.0012041
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0005483, 0.0005505

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001064, upper bound: 0.0001038
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001097, upper bound: 0.0001004
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006596, 0.0006570
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000909, 0.0000910
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0008087, 0.0008049
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002229, 0.0002243
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001189, 0.0001194
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0005018, 0.0004998
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000398, 0.0000399
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0013418, 0.0013444
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0012041, 0.0012091
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0005502, 0.0005486

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001045, upper bound: 0.0001063
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001074, upper bound: 0.0001034
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006570, 0.0006596
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000910, 0.0000909
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0008049, 0.0008087
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002243, 0.0002229
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001194, 0.0001189
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004998, 0.0005018
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000399, 0.0000398
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0013444, 0.0013418
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0012091, 0.0012041
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0005486, 0.0005502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001033, upper bound: 0.0001074
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001063, upper bound: 0.0001045
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006595, 0.0006571
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000910, 0.0000908
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0008086, 0.0008048
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002227, 0.0002245
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001189, 0.0001195
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0005018, 0.0004998
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000397, 0.0000399
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0013427, 0.0013434
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0012041, 0.0012096
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0005505, 0.0005483

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001004, upper bound: 0.0001097
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001038, upper bound: 0.0001064
time: 0.61 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.63 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 3, lower bound: -0.0001064, upper bound: 0.0001038
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 3, lower bound: -0.0001097, upper bound: 0.0001004
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 3, lower bound: -0.0001045, upper bound: 0.0001063
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 3, lower bound: -0.0001074, upper bound: 0.0001034
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 3, lower bound: -0.0001033, upper bound: 0.0001074
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 3, lower bound: -0.0001063, upper bound: 0.0001045
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 3, lower bound: -0.0001004, upper bound: 0.0001097
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 3, lower bound: -0.0001038, upper bound: 0.0001064

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006331, 0.0006229
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000825, 0.0000803
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007728, 0.0007596
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002108, 0.0002104
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001115, 0.0001136
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004814, 0.0004736
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000398, 0.0000394
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012922, 0.0013147
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0011225, 0.0011463
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0005185, 0.0005067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000892, upper bound: 0.0000902
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000892, upper bound: 0.0000902
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006204, 0.0006362
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000801, 0.0000827
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007558, 0.0007769
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002124, 0.0002090
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001142, 0.0001109
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004716, 0.0004838
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000396, 0.0000396
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0013142, 0.0012914
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0011511, 0.0011169
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0005046, 0.0005205

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000928, upper bound: 0.0000867
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000928, upper bound: 0.0000867
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006360, 0.0006203
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000826, 0.0000802
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007765, 0.0007559
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002092, 0.0002121
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001109, 0.0001141
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004836, 0.0004716
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000396, 0.0000395
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012906, 0.0013142
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0011170, 0.0011503
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0005200, 0.0005048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000892, upper bound: 0.0000904
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000892, upper bound: 0.0000904
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006230, 0.0006336
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000802, 0.0000827
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007597, 0.0007737
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002106, 0.0002106
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001138, 0.0001114
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004736, 0.0004818
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000395, 0.0000397
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0013142, 0.0012931
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0011477, 0.0011219
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0005065, 0.0005192

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000927, upper bound: 0.0000869
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000927, upper bound: 0.0000869
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006336, 0.0006230
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000827, 0.0000802
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007737, 0.0007597
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002106, 0.0002106
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001114, 0.0001138
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004818, 0.0004736
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000397, 0.0000395
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012931, 0.0013142
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0011219, 0.0011477
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0005192, 0.0005065

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000869, upper bound: 0.0000927
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000869, upper bound: 0.0000927
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006203, 0.0006360
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000802, 0.0000826
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007559, 0.0007765
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002121, 0.0002092
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001141, 0.0001109
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004716, 0.0004836
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000395, 0.0000396
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0013142, 0.0012906
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0011503, 0.0011170
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0005048, 0.0005200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000904, upper bound: 0.0000892
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000904, upper bound: 0.0000892
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006362, 0.0006204
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000827, 0.0000801
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007769, 0.0007558
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002090, 0.0002124
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001109, 0.0001142
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004838, 0.0004716
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000396, 0.0000396
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012914, 0.0013142
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0011169, 0.0011511
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0005205, 0.0005046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000867, upper bound: 0.0000928
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000867, upper bound: 0.0000928
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006229, 0.0006331
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000803, 0.0000825
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007596, 0.0007728
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002104, 0.0002108
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001136, 0.0001115
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004736, 0.0004814
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000394, 0.0000398
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0013147, 0.0012922
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0011463, 0.0011225
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0005067, 0.0005185

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000902, upper bound: 0.0000892
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000902, upper bound: 0.0000892
time: 0.56 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.56 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 3, lower bound: -0.0000892, upper bound: 0.0000902
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 3, lower bound: -0.0000892, upper bound: 0.0000902
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 3, lower bound: -0.0000928, upper bound: 0.0000867
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 3, lower bound: -0.0000928, upper bound: 0.0000867
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 3, lower bound: -0.0000892, upper bound: 0.0000904
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 3, lower bound: -0.0000892, upper bound: 0.0000904
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 3, lower bound: -0.0000927, upper bound: 0.0000869
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 3, lower bound: -0.0000927, upper bound: 0.0000869
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 3, lower bound: -0.0000869, upper bound: 0.0000927
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 3, lower bound: -0.0000869, upper bound: 0.0000927
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 3, lower bound: -0.0000904, upper bound: 0.0000892
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 3, lower bound: -0.0000904, upper bound: 0.0000892
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 3, lower bound: -0.0000867, upper bound: 0.0000928
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 3, lower bound: -0.0000867, upper bound: 0.0000928
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 3, lower bound: -0.0000902, upper bound: 0.0000892
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 3, lower bound: -0.0000902, upper bound: 0.0000892

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006176, 0.0006169
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000803, 0.0000793
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007532, 0.0007519
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002106, 0.0002100
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001103, 0.0001106
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004695, 0.0004690
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000391, 0.0000387
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012754, 0.0012862
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0011096, 0.0011155
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0005044, 0.0005005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000890, upper bound: 0.0000899
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000890, upper bound: 0.0000896
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006331, 0.0006074
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000825, 0.0000781
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007728, 0.0007399
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002104, 0.0002104
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001085, 0.0001136
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004814, 0.0004617
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000398, 0.0000388
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012637, 0.0013147
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0010917, 0.0011463
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0005185, 0.0004926

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000890, upper bound: 0.0000899
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000890, upper bound: 0.0000896
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006049, 0.0006284
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000779, 0.0000813
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007362, 0.0007671
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002122, 0.0002086
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001127, 0.0001079
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004598, 0.0004779
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000390, 0.0000388
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012963, 0.0012629
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0011361, 0.0010862
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0004904, 0.0005134

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000926, upper bound: 0.0000865
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000926, upper bound: 0.0000865
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006204, 0.0006207
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000801, 0.0000805
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007558, 0.0007572
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002120, 0.0002090
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001112, 0.0001109
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004716, 0.0004719
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000396, 0.0000389
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012857, 0.0012914
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0011203, 0.0011169
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0005046, 0.0005064

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000926, upper bound: 0.0000865
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000926, upper bound: 0.0000865
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006205, 0.0006132
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000804, 0.0000792
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007569, 0.0007476
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002091, 0.0002117
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001097, 0.0001111
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004718, 0.0004662
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000390, 0.0000390
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012758, 0.0012858
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0011043, 0.0011195
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0005058, 0.0004984

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000890, upper bound: 0.0000902
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000890, upper bound: 0.0000902
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006360, 0.0006048
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000826, 0.0000780
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007765, 0.0007362
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002088, 0.0002121
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001079, 0.0001141
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004836, 0.0004597
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000396, 0.0000389
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012621, 0.0013142
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0010862, 0.0011503
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0005200, 0.0004907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000890, upper bound: 0.0000902
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000890, upper bound: 0.0000902
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006075, 0.0006238
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000780, 0.0000812
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007400, 0.0007614
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002104, 0.0002102
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001119, 0.0001084
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004618, 0.0004743
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000388, 0.0000392
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012968, 0.0012646
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0011291, 0.0010912
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0004924, 0.0005108

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000924, upper bound: 0.0000867
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000925, upper bound: 0.0000866
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006230, 0.0006181
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000802, 0.0000805
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007597, 0.0007541
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002102, 0.0002106
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001108, 0.0001114
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004736, 0.0004700
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000395, 0.0000391
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012857, 0.0012931
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0011170, 0.0011219
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0005065, 0.0005051

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000924, upper bound: 0.0000867
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000925, upper bound: 0.0000866
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006181, 0.0006166
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000805, 0.0000792
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007541, 0.0007517
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002104, 0.0002102
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001102, 0.0001108
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004700, 0.0004688
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000391, 0.0000388
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012763, 0.0012857
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0011096, 0.0011170
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0005051, 0.0005004

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000866, upper bound: 0.0000925
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000867, upper bound: 0.0000924
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006336, 0.0006075
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000827, 0.0000780
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007737, 0.0007400
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002102, 0.0002106
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001084, 0.0001138
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004818, 0.0004618
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000397, 0.0000388
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012646, 0.0013142
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0010912, 0.0011477
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0005192, 0.0004924

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000866, upper bound: 0.0000925
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000867, upper bound: 0.0000924
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006048, 0.0006283
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000780, 0.0000812
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007362, 0.0007670
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002119, 0.0002088
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001127, 0.0001079
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004597, 0.0004778
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000389, 0.0000389
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012964, 0.0012621
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0011360, 0.0010862
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0004907, 0.0005133

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000902, upper bound: 0.0000890
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000902, upper bound: 0.0000890
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006203, 0.0006205
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000802, 0.0000804
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007559, 0.0007569
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002117, 0.0002092
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001111, 0.0001109
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004716, 0.0004718
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000395, 0.0000390
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012858, 0.0012906
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0011195, 0.0011170
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0005048, 0.0005058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000902, upper bound: 0.0000890
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000902, upper bound: 0.0000890
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006207, 0.0006128
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000805, 0.0000791
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007572, 0.0007471
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002089, 0.0002120
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001096, 0.0001112
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004719, 0.0004659
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000389, 0.0000391
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012760, 0.0012857
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0011041, 0.0011203
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0005064, 0.0004983

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000865, upper bound: 0.0000926
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000865, upper bound: 0.0000926
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006362, 0.0006049
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000827, 0.0000779
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007769, 0.0007362
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002086, 0.0002124
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001079, 0.0001142
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004838, 0.0004598
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000396, 0.0000390
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012629, 0.0013142
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0010862, 0.0011511
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0005205, 0.0004904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000865, upper bound: 0.0000926
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000865, upper bound: 0.0000926
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006074, 0.0006237
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000781, 0.0000811
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007399, 0.0007611
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002102, 0.0002104
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001119, 0.0001085
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004617, 0.0004742
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000388, 0.0000392
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012974, 0.0012637
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0011288, 0.0010917
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0004926, 0.0005107

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000896, upper bound: 0.0000890
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000899, upper bound: 0.0000890
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006229, 0.0006176
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000803, 0.0000803
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007596, 0.0007532
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002100, 0.0002108
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001106, 0.0001115
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004736, 0.0004695
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000394, 0.0000391
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012862, 0.0012922
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0011155, 0.0011225
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0005067, 0.0005044

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000896, upper bound: 0.0000890
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000899, upper bound: 0.0000890
time: 0.57 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.75 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0000890, upper bound: 0.0000899
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0000890, upper bound: 0.0000896
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0000890, upper bound: 0.0000899
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0000890, upper bound: 0.0000896
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0000926, upper bound: 0.0000865
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0000926, upper bound: 0.0000865
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0000926, upper bound: 0.0000865
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0000926, upper bound: 0.0000865
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0000890, upper bound: 0.0000902
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0000890, upper bound: 0.0000902
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0000890, upper bound: 0.0000902
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0000890, upper bound: 0.0000902
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0000924, upper bound: 0.0000867
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0000925, upper bound: 0.0000866
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0000924, upper bound: 0.0000867
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0000925, upper bound: 0.0000866
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0000866, upper bound: 0.0000925
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0000867, upper bound: 0.0000924
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0000866, upper bound: 0.0000925
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0000867, upper bound: 0.0000924
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0000902, upper bound: 0.0000890
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0000902, upper bound: 0.0000890
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0000902, upper bound: 0.0000890
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0000902, upper bound: 0.0000890
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0000865, upper bound: 0.0000926
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0000865, upper bound: 0.0000926
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0000865, upper bound: 0.0000926
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0000865, upper bound: 0.0000926
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0000896, upper bound: 0.0000890
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0000899, upper bound: 0.0000890
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0000896, upper bound: 0.0000890
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0000899, upper bound: 0.0000890

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006110, 0.0006106
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000758, 0.0000748
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007435, 0.0007427
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002097, 0.0002090
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001085, 0.0001088
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004644, 0.0004641
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000392, 0.0000387
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012692, 0.0012791
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0010896, 0.0010947
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0004927, 0.0004894

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000719, upper bound: 0.0000741
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000720, upper bound: 0.0000745
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006115, 0.0006103
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000759, 0.0000747
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007443, 0.0007423
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002097, 0.0002088
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001085, 0.0001089
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004648, 0.0004638
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000392, 0.0000387
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012682, 0.0012798
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0010888, 0.0010953
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0004930, 0.0004889

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000719, upper bound: 0.0000738
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000720, upper bound: 0.0000741
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006266, 0.0006014
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000782, 0.0000736
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007631, 0.0007310
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002094, 0.0002094
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001068, 0.0001118
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004763, 0.0004571
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000398, 0.0000388
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012573, 0.0013081
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0010722, 0.0011252
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0005069, 0.0004816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.25 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000719, upper bound: 0.0000741
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000720, upper bound: 0.0000745
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006272, 0.0006007
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000783, 0.0000736
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007639, 0.0007303
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002094, 0.0002092
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001067, 0.0001119
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004767, 0.0004565
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000398, 0.0000388
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012565, 0.0013088
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0010709, 0.0011258
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0005071, 0.0004809

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.26 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000719, upper bound: 0.0000738
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000720, upper bound: 0.0000741
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0005983, 0.0006224
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000734, 0.0000769
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007266, 0.0007580
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002111, 0.0002077
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001110, 0.0001061
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004546, 0.0004731
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000390, 0.0000388
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012899, 0.0012557
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0011160, 0.0010653
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0004788, 0.0005021

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.27 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000767, upper bound: 0.0000699
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000766, upper bound: 0.0000699
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0005990, 0.0006218
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000734, 0.0000768
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007275, 0.0007575
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002112, 0.0002077
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001109, 0.0001062
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004552, 0.0004727
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000390, 0.0000388
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012892, 0.0012564
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0011153, 0.0010668
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0004793, 0.0005017

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.26 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000767, upper bound: 0.0000699
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000766, upper bound: 0.0000699
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006139, 0.0006147
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000758, 0.0000761
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007462, 0.0007483
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002109, 0.0002081
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001095, 0.0001091
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004665, 0.0004673
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000396, 0.0000389
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012793, 0.0012848
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0011000, 0.0010958
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0004929, 0.0004949

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.25 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000767, upper bound: 0.0000699
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000766, upper bound: 0.0000699
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006147, 0.0006140
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000758, 0.0000760
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007471, 0.0007476
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002110, 0.0002081
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001094, 0.0001092
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004671, 0.0004668
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000396, 0.0000389
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012785, 0.0012854
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0010995, 0.0010973
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0004935, 0.0004947

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.26 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000767, upper bound: 0.0000699
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000766, upper bound: 0.0000699
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006139, 0.0006072
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000759, 0.0000747
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007473, 0.0007387
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002082, 0.0002108
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001080, 0.0001093
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004666, 0.0004615
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000390, 0.0000390
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012700, 0.0012786
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0010843, 0.0010987
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0004942, 0.0004872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.26 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000719, upper bound: 0.0000743
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000720, upper bound: 0.0000746
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006146, 0.0006066
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000759, 0.0000746
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007480, 0.0007379
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002081, 0.0002107
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001079, 0.0001094
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004672, 0.0004610
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000390, 0.0000390
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012686, 0.0012795
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0010834, 0.0010998
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0004946, 0.0004867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.26 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000719, upper bound: 0.0000742
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000720, upper bound: 0.0000745
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006295, 0.0005989
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000783, 0.0000735
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007669, 0.0007276
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002079, 0.0002112
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001062, 0.0001123
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004785, 0.0004551
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000396, 0.0000389
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012555, 0.0013076
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0010670, 0.0011292
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0005083, 0.0004795

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.34 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000719, upper bound: 0.0000743
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000720, upper bound: 0.0000746
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006303, 0.0005982
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000783, 0.0000735
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007676, 0.0007266
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002079, 0.0002111
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001061, 0.0001124
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004791, 0.0004546
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000396, 0.0000389
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012549, 0.0013085
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0010654, 0.0011303
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0005088, 0.0004790

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.26 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000719, upper bound: 0.0000742
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000720, upper bound: 0.0000745
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006008, 0.0006177
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000735, 0.0000768
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007304, 0.0007523
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002092, 0.0002092
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001102, 0.0001066
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004566, 0.0004695
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000388, 0.0000392
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012908, 0.0012575
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0011091, 0.0010704
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0004807, 0.0004996

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000766, upper bound: 0.0000700
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000765, upper bound: 0.0000700
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006015, 0.0006172
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000735, 0.0000767
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007311, 0.0007518
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002094, 0.0002092
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001101, 0.0001068
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004571, 0.0004692
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000388, 0.0000392
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012896, 0.0012585
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0011083, 0.0010716
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0004813, 0.0004992

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000767, upper bound: 0.0000700
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000766, upper bound: 0.0000700
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006165, 0.0006118
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000758, 0.0000760
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007500, 0.0007449
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002091, 0.0002096
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001091, 0.0001096
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004686, 0.0004651
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000395, 0.0000391
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012790, 0.0012865
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0010968, 0.0011009
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0004949, 0.0004936

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.27 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000766, upper bound: 0.0000700
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000765, upper bound: 0.0000700
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006172, 0.0006115
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000759, 0.0000760
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007507, 0.0007444
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002092, 0.0002096
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001090, 0.0001097
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004691, 0.0004648
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000395, 0.0000391
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012785, 0.0012875
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0010962, 0.0011022
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0004955, 0.0004934

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.27 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000767, upper bound: 0.0000700
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000766, upper bound: 0.0000700
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006115, 0.0006103
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000760, 0.0000747
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007444, 0.0007424
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002095, 0.0002092
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001085, 0.0001090
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004648, 0.0004638
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000391, 0.0000388
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012701, 0.0012785
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0010894, 0.0010962
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0004934, 0.0004891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.26 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000700, upper bound: 0.0000766
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000700, upper bound: 0.0000767
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006118, 0.0006100
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000760, 0.0000747
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007449, 0.0007421
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002095, 0.0002091
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001085, 0.0001091
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004651, 0.0004636
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000391, 0.0000388
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012692, 0.0012790
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0010887, 0.0010968
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0004936, 0.0004887

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.25 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000700, upper bound: 0.0000765
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000700, upper bound: 0.0000766
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006271, 0.0006015
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000783, 0.0000735
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007640, 0.0007311
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002092, 0.0002096
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001068, 0.0001120
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004767, 0.0004571
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000397, 0.0000388
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012585, 0.0013076
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0010716, 0.0011267
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0005076, 0.0004813

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.26 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000700, upper bound: 0.0000766
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000700, upper bound: 0.0000767
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006275, 0.0006008
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000784, 0.0000735
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007645, 0.0007304
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002092, 0.0002094
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001066, 0.0001120
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004770, 0.0004566
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000397, 0.0000388
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012575, 0.0013080
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0010704, 0.0011274
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0005078, 0.0004807

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.25 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000700, upper bound: 0.0000765
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000700, upper bound: 0.0000766
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0005982, 0.0006223
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000735, 0.0000768
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007266, 0.0007578
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002109, 0.0002079
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001110, 0.0001061
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004546, 0.0004730
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000389, 0.0000389
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012901, 0.0012549
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0011159, 0.0010654
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0004790, 0.0005020

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.26 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000745, upper bound: 0.0000720
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000742, upper bound: 0.0000719
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0005989, 0.0006216
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000735, 0.0000767
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007276, 0.0007574
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002110, 0.0002079
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001109, 0.0001062
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004551, 0.0004726
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000389, 0.0000389
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012892, 0.0012555
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0011151, 0.0010670
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0004795, 0.0005016

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000746, upper bound: 0.0000720
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000743, upper bound: 0.0000719
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006138, 0.0006146
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000759, 0.0000759
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007462, 0.0007480
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002107, 0.0002083
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001094, 0.0001090
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004665, 0.0004672
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000396, 0.0000390
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012795, 0.0012839
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0010998, 0.0010959
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0004932, 0.0004946

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.26 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000745, upper bound: 0.0000720
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000742, upper bound: 0.0000719
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006146, 0.0006139
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000759, 0.0000759
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007472, 0.0007473
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002108, 0.0002083
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001093, 0.0001092
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004671, 0.0004666
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000396, 0.0000390
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012786, 0.0012845
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0010987, 0.0010975
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0004937, 0.0004942

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.26 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000746, upper bound: 0.0000720
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000743, upper bound: 0.0000719
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006140, 0.0006068
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000760, 0.0000746
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007476, 0.0007381
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002079, 0.0002110
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001079, 0.0001094
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004668, 0.0004612
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000389, 0.0000391
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012701, 0.0012785
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0010841, 0.0010995
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0004947, 0.0004871

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.27 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000699, upper bound: 0.0000766
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000699, upper bound: 0.0000767
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006147, 0.0006061
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000761, 0.0000746
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007483, 0.0007374
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002079, 0.0002109
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001078, 0.0001095
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004673, 0.0004607
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000389, 0.0000391
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012688, 0.0012793
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0010833, 0.0011000
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0004949, 0.0004867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000699, upper bound: 0.0000766
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000699, upper bound: 0.0000767
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006297, 0.0005990
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000784, 0.0000734
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007672, 0.0007275
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002077, 0.0002114
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001062, 0.0001123
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004787, 0.0004552
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000396, 0.0000390
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012564, 0.0013075
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0010668, 0.0011300
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0005089, 0.0004793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000699, upper bound: 0.0000766
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000699, upper bound: 0.0000767
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006304, 0.0005983
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000785, 0.0000734
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007679, 0.0007266
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002077, 0.0002113
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001061, 0.0001124
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004792, 0.0004546
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000396, 0.0000390
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012557, 0.0013083
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0010653, 0.0011305
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0005090, 0.0004788

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.27 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000699, upper bound: 0.0000766
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000699, upper bound: 0.0000767
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006007, 0.0006176
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000736, 0.0000767
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007303, 0.0007519
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002090, 0.0002094
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001102, 0.0001067
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004565, 0.0004694
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000388, 0.0000392
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012914, 0.0012565
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0011088, 0.0010709
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0004809, 0.0004994

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000741, upper bound: 0.0000720
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000738, upper bound: 0.0000719
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006014, 0.0006171
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000736, 0.0000766
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007310, 0.0007515
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002092, 0.0002094
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001101, 0.0001068
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004571, 0.0004690
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000388, 0.0000392
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012902, 0.0012573
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0011080, 0.0010722
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0004816, 0.0004990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000745, upper bound: 0.0000720
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000741, upper bound: 0.0000719
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006164, 0.0006115
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000760, 0.0000759
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007499, 0.0007443
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002088, 0.0002098
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001089, 0.0001096
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004685, 0.0004648
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000394, 0.0000392
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012798, 0.0012855
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0010953, 0.0011014
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0004951, 0.0004930

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000741, upper bound: 0.0000720
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000738, upper bound: 0.0000719
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037685, -0.0023604, -0.0037685, -0.0023604, -0.0006171, 0.0006110
1: -0.0045767, -0.0043145, -0.0045767, -0.0043145, -0.0000760, 0.0000758
2: 0.0097732, 0.0115660, 0.0097732, 0.0115660, -0.0007506, 0.0007435
3: 1.0086166, 1.0090152, 1.0086166, 1.0090152, -0.0002090, 0.0002098
4: -0.0034998, -0.0032222, -0.0034998, -0.0032222, -0.0001088, 0.0001098
5: 0.0010670, 0.0021438, 0.0010670, 0.0021438, -0.0004691, 0.0004644
6: -0.0025334, -0.0024818, -0.0025334, -0.0024818, -0.0000394, 0.0000392
7: -0.0096087, -0.0071702, -0.0096087, -0.0071702, -0.0012791, 0.0012863
8: -0.0054149, -0.0025123, -0.0054149, -0.0025123, -0.0010947, 0.0011027
9: -0.0029352, -0.0015587, -0.0029352, -0.0015587, -0.0004958, 0.0004927

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.34 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000745, upper bound: 0.0000720
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000741, upper bound: 0.0000719
time: 0.68 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.07 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000719, upper bound: 0.0000741
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000720, upper bound: 0.0000745
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000719, upper bound: 0.0000738
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000720, upper bound: 0.0000741
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000719, upper bound: 0.0000741
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000720, upper bound: 0.0000745
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000719, upper bound: 0.0000738
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000720, upper bound: 0.0000741
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000767, upper bound: 0.0000699
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000766, upper bound: 0.0000699
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000767, upper bound: 0.0000699
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000766, upper bound: 0.0000699
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000767, upper bound: 0.0000699
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000766, upper bound: 0.0000699
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000767, upper bound: 0.0000699
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000766, upper bound: 0.0000699
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000719, upper bound: 0.0000743
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000720, upper bound: 0.0000746
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000719, upper bound: 0.0000742
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000720, upper bound: 0.0000745
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000719, upper bound: 0.0000743
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000720, upper bound: 0.0000746
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000719, upper bound: 0.0000742
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000720, upper bound: 0.0000745
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000766, upper bound: 0.0000700
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000765, upper bound: 0.0000700
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000767, upper bound: 0.0000700
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000766, upper bound: 0.0000700
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000766, upper bound: 0.0000700
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000765, upper bound: 0.0000700
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000767, upper bound: 0.0000700
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000766, upper bound: 0.0000700
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000700, upper bound: 0.0000766
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000700, upper bound: 0.0000767
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000700, upper bound: 0.0000765
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000700, upper bound: 0.0000766
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000700, upper bound: 0.0000766
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000700, upper bound: 0.0000767
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000700, upper bound: 0.0000765
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000700, upper bound: 0.0000766
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000745, upper bound: 0.0000720
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000742, upper bound: 0.0000719
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000746, upper bound: 0.0000720
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000743, upper bound: 0.0000719
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000745, upper bound: 0.0000720
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000742, upper bound: 0.0000719
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000746, upper bound: 0.0000720
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000743, upper bound: 0.0000719
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000699, upper bound: 0.0000766
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000699, upper bound: 0.0000767
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000699, upper bound: 0.0000766
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000699, upper bound: 0.0000767
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000699, upper bound: 0.0000766
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000699, upper bound: 0.0000767
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000699, upper bound: 0.0000766
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000699, upper bound: 0.0000767
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000741, upper bound: 0.0000720
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000738, upper bound: 0.0000719
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000745, upper bound: 0.0000720
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000741, upper bound: 0.0000719
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000741, upper bound: 0.0000720
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000738, upper bound: 0.0000719
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000745, upper bound: 0.0000720
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -0.0000741, upper bound: 0.0000719

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.77 + 172.27 = 175.04 seconds
