## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0010557


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0006495, 0.0014115, 0.0006495, 0.0014115, -0.0007537, 0.0007537)
1: (0.9929024, 0.9948239, 0.9929024, 0.9948239, -0.0018200, 0.0018200)
2: (-0.0070641, -0.0046181, -0.0070641, -0.0046181, -0.0024459, 0.0024459)
3: (0.0034906, 0.0042876, 0.0034906, 0.0042876, -0.0007283, 0.0007283)
4: (0.0023821, 0.0040001, 0.0023821, 0.0040001, -0.0016180, 0.0016180)
5: (0.0052827, 0.0071503, 0.0052827, 0.0071503, -0.0018676, 0.0018676)
6: (-0.0015731, -0.0007045, -0.0015731, -0.0007045, -0.0008686, 0.0008686)
7: (-0.0087607, -0.0074160, -0.0087607, -0.0074160, -0.0013447, 0.0013447)
8: (0.0032780, 0.0075329, 0.0032780, 0.0075329, -0.0038108, 0.0038108)
9: (-0.0046987, -0.0022600, -0.0046987, -0.0022600, -0.0024387, 0.0024387)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.13 + 1.75 = 2.88 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0010955, upper bound: 0.0010955

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010762, upper bound: 0.0010897
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010897, upper bound: 0.0010762
time: 0.79 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.75 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.75
Output dim: 1, lower bound: -0.0010762, upper bound: 0.0010897
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.75
Output dim: 1, lower bound: -0.0010897, upper bound: 0.0010762

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006495, 0.0014115, 0.0006495, 0.0014115, -0.0007534, 0.0007532
1: 0.9929024, 0.9948239, 0.9929024, 0.9948239, -0.0018135, 0.0018161
2: -0.0070641, -0.0046181, -0.0070641, -0.0046181, -0.0024459, 0.0024459
3: 0.0034906, 0.0042876, 0.0034906, 0.0042876, -0.0007257, 0.0007239
4: 0.0023821, 0.0040001, 0.0023821, 0.0040001, -0.0016180, 0.0016180
5: 0.0052827, 0.0071503, 0.0052827, 0.0071503, -0.0018676, 0.0018676
6: -0.0015731, -0.0007045, -0.0015731, -0.0007045, -0.0008686, 0.0008686
7: -0.0087607, -0.0074160, -0.0087607, -0.0074160, -0.0013447, 0.0013447
8: 0.0032780, 0.0075329, 0.0032780, 0.0075329, -0.0038110, 0.0037750
9: -0.0046987, -0.0022600, -0.0046987, -0.0022600, -0.0024387, 0.0024387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010610, upper bound: 0.0010750
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010614, upper bound: 0.0010743
time: 0.97 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006495, 0.0014115, 0.0006495, 0.0014115, -0.0007532, 0.0007534
1: 0.9929024, 0.9948239, 0.9929024, 0.9948239, -0.0018161, 0.0018135
2: -0.0070641, -0.0046181, -0.0070641, -0.0046181, -0.0024459, 0.0024459
3: 0.0034906, 0.0042876, 0.0034906, 0.0042876, -0.0007239, 0.0007257
4: 0.0023821, 0.0040001, 0.0023821, 0.0040001, -0.0016180, 0.0016180
5: 0.0052827, 0.0071503, 0.0052827, 0.0071503, -0.0018676, 0.0018676
6: -0.0015731, -0.0007045, -0.0015731, -0.0007045, -0.0008686, 0.0008686
7: -0.0087607, -0.0074160, -0.0087607, -0.0074160, -0.0013447, 0.0013447
8: 0.0032780, 0.0075329, 0.0032780, 0.0075329, -0.0037750, 0.0038110
9: -0.0046987, -0.0022600, -0.0046987, -0.0022600, -0.0024387, 0.0024387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010743, upper bound: 0.0010614
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010751, upper bound: 0.0010610
time: 0.89 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.80 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.80
Output dim: 1, lower bound: -0.0010610, upper bound: 0.0010750
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.80
Output dim: 1, lower bound: -0.0010614, upper bound: 0.0010743
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.80
Output dim: 1, lower bound: -0.0010743, upper bound: 0.0010614
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.80
Output dim: 1, lower bound: -0.0010751, upper bound: 0.0010610

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006495, 0.0014115, 0.0006495, 0.0014115, -0.0007532, 0.0007529
1: 0.9929024, 0.9948239, 0.9929024, 0.9948239, -0.0018093, 0.0018131
2: -0.0070641, -0.0046181, -0.0070641, -0.0046181, -0.0024459, 0.0024459
3: 0.0034906, 0.0042876, 0.0034906, 0.0042876, -0.0007236, 0.0007209
4: 0.0023821, 0.0040001, 0.0023821, 0.0040001, -0.0016180, 0.0016180
5: 0.0052827, 0.0071503, 0.0052827, 0.0071503, -0.0018676, 0.0018676
6: -0.0015731, -0.0007045, -0.0015731, -0.0007045, -0.0008686, 0.0008686
7: -0.0087607, -0.0074160, -0.0087607, -0.0074160, -0.0013447, 0.0013447
8: 0.0032780, 0.0075329, 0.0032780, 0.0075329, -0.0037082, 0.0036555
9: -0.0046987, -0.0022600, -0.0046987, -0.0022600, -0.0024387, 0.0024387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010361, upper bound: 0.0010442
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010361, upper bound: 0.0010442
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006495, 0.0014115, 0.0006495, 0.0014115, -0.0007531, 0.0007530
1: 0.9929024, 0.9948239, 0.9929024, 0.9948239, -0.0018105, 0.0018119
2: -0.0070641, -0.0046181, -0.0070641, -0.0046181, -0.0024459, 0.0024459
3: 0.0034906, 0.0042876, 0.0034906, 0.0042876, -0.0007227, 0.0007218
4: 0.0023821, 0.0040001, 0.0023821, 0.0040001, -0.0016180, 0.0016180
5: 0.0052827, 0.0071503, 0.0052827, 0.0071503, -0.0018676, 0.0018676
6: -0.0015731, -0.0007045, -0.0015731, -0.0007045, -0.0008686, 0.0008686
7: -0.0087607, -0.0074160, -0.0087607, -0.0074160, -0.0013447, 0.0013447
8: 0.0032780, 0.0075329, 0.0032780, 0.0075329, -0.0036911, 0.0036722
9: -0.0046987, -0.0022600, -0.0046987, -0.0022600, -0.0024387, 0.0024387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010368, upper bound: 0.0010435
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010368, upper bound: 0.0010435
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006495, 0.0014115, 0.0006495, 0.0014115, -0.0007530, 0.0007531
1: 0.9929024, 0.9948239, 0.9929024, 0.9948239, -0.0018119, 0.0018105
2: -0.0070641, -0.0046181, -0.0070641, -0.0046181, -0.0024459, 0.0024459
3: 0.0034906, 0.0042876, 0.0034906, 0.0042876, -0.0007218, 0.0007227
4: 0.0023821, 0.0040001, 0.0023821, 0.0040001, -0.0016180, 0.0016180
5: 0.0052827, 0.0071503, 0.0052827, 0.0071503, -0.0018676, 0.0018676
6: -0.0015731, -0.0007045, -0.0015731, -0.0007045, -0.0008686, 0.0008686
7: -0.0087607, -0.0074160, -0.0087607, -0.0074160, -0.0013447, 0.0013447
8: 0.0032780, 0.0075329, 0.0032780, 0.0075329, -0.0036722, 0.0036911
9: -0.0046987, -0.0022600, -0.0046987, -0.0022600, -0.0024387, 0.0024387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010435, upper bound: 0.0010368
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010435, upper bound: 0.0010368
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006495, 0.0014115, 0.0006495, 0.0014115, -0.0007529, 0.0007532
1: 0.9929024, 0.9948239, 0.9929024, 0.9948239, -0.0018131, 0.0018093
2: -0.0070641, -0.0046181, -0.0070641, -0.0046181, -0.0024459, 0.0024459
3: 0.0034906, 0.0042876, 0.0034906, 0.0042876, -0.0007209, 0.0007236
4: 0.0023821, 0.0040001, 0.0023821, 0.0040001, -0.0016180, 0.0016180
5: 0.0052827, 0.0071503, 0.0052827, 0.0071503, -0.0018676, 0.0018676
6: -0.0015731, -0.0007045, -0.0015731, -0.0007045, -0.0008686, 0.0008686
7: -0.0087607, -0.0074160, -0.0087607, -0.0074160, -0.0013447, 0.0013447
8: 0.0032780, 0.0075329, 0.0032780, 0.0075329, -0.0036555, 0.0037082
9: -0.0046987, -0.0022600, -0.0046987, -0.0022600, -0.0024387, 0.0024387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010442, upper bound: 0.0010361
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010442, upper bound: 0.0010361
time: 0.85 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.81 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 1, lower bound: -0.0010361, upper bound: 0.0010442
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 1, lower bound: -0.0010361, upper bound: 0.0010442
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 1, lower bound: -0.0010368, upper bound: 0.0010435
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 1, lower bound: -0.0010368, upper bound: 0.0010435
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 1, lower bound: -0.0010435, upper bound: 0.0010368
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 1, lower bound: -0.0010435, upper bound: 0.0010368
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 1, lower bound: -0.0010442, upper bound: 0.0010361
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 1, lower bound: -0.0010442, upper bound: 0.0010361

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.88 + 19.66 = 22.54 seconds
