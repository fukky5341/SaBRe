## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0004916


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0001601, 0.0006554, 0.0001601, 0.0006554, -0.0003564, 0.0003564)
1: (0.9944912, 0.9955402, 0.9944912, 0.9955402, -0.0007547, 0.0007547)
2: (-0.0079258, -0.0076780, -0.0079258, -0.0076780, -0.0001783, 0.0001783)
3: (0.0028884, 0.0035081, 0.0028884, 0.0035081, -0.0004459, 0.0004459)
4: (0.0027627, 0.0035705, 0.0027627, 0.0035705, -0.0005812, 0.0005812)
5: (0.0037417, 0.0049155, 0.0037417, 0.0049155, -0.0008445, 0.0008445)
6: (-0.0007183, 0.0003671, -0.0007183, 0.0003671, -0.0007809, 0.0007809)
7: (-0.0074047, -0.0069026, -0.0074047, -0.0069026, -0.0003613, 0.0003613)
8: (0.0081031, 0.0081692, 0.0081031, 0.0081692, -0.0000476, 0.0000476)
9: (-0.0030091, -0.0022922, -0.0030091, -0.0022922, -0.0005159, 0.0005159)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.29 + 1.23 = 2.52 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0005312, upper bound: 0.0005312

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005251, upper bound: 0.0005247
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005247, upper bound: 0.0005251
time: 0.42 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.01 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.01
Output dim: 1, lower bound: -0.0005251, upper bound: 0.0005247
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.01
Output dim: 1, lower bound: -0.0005247, upper bound: 0.0005251

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0001601, 0.0006554, 0.0001601, 0.0006554, -0.0003555, 0.0003560
1: 0.9944912, 0.9955402, 0.9944912, 0.9955402, -0.0007538, 0.0007529
2: -0.0079258, -0.0076780, -0.0079258, -0.0076780, -0.0001778, 0.0001781
3: 0.0028884, 0.0035081, 0.0028884, 0.0035081, -0.0004448, 0.0004453
4: 0.0027627, 0.0035705, 0.0027627, 0.0035705, -0.0005798, 0.0005805
5: 0.0037417, 0.0049155, 0.0037417, 0.0049155, -0.0008425, 0.0008435
6: -0.0007183, 0.0003671, -0.0007183, 0.0003671, -0.0007800, 0.0007791
7: -0.0074047, -0.0069026, -0.0074047, -0.0069026, -0.0003608, 0.0003604
8: 0.0081031, 0.0081692, 0.0081031, 0.0081692, -0.0000475, 0.0000475
9: -0.0030091, -0.0022922, -0.0030091, -0.0022922, -0.0005153, 0.0005146

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005142, upper bound: 0.0004827
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0004830, upper bound: 0.0005140
time: 0.43 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0001601, 0.0006554, 0.0001601, 0.0006554, -0.0003560, 0.0003555
1: 0.9944912, 0.9955402, 0.9944912, 0.9955402, -0.0007529, 0.0007538
2: -0.0079258, -0.0076780, -0.0079258, -0.0076780, -0.0001781, 0.0001778
3: 0.0028884, 0.0035081, 0.0028884, 0.0035081, -0.0004453, 0.0004448
4: 0.0027627, 0.0035705, 0.0027627, 0.0035705, -0.0005805, 0.0005798
5: 0.0037417, 0.0049155, 0.0037417, 0.0049155, -0.0008435, 0.0008425
6: -0.0007183, 0.0003671, -0.0007183, 0.0003671, -0.0007791, 0.0007800
7: -0.0074047, -0.0069026, -0.0074047, -0.0069026, -0.0003604, 0.0003608
8: 0.0081031, 0.0081692, 0.0081031, 0.0081692, -0.0000475, 0.0000475
9: -0.0030091, -0.0022922, -0.0030091, -0.0022922, -0.0005146, 0.0005153

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005140, upper bound: 0.0004830
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0004827, upper bound: 0.0005142
time: 0.42 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.16 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 1, lower bound: -0.0005142, upper bound: 0.0004827
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 1, lower bound: -0.0004830, upper bound: 0.0005140
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 1, lower bound: -0.0005140, upper bound: 0.0004830
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 1, lower bound: -0.0004827, upper bound: 0.0005142

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0001601, 0.0006554, 0.0001601, 0.0006554, -0.0003394, 0.0003488
1: 0.9944912, 0.9955402, 0.9944912, 0.9955402, -0.0007387, 0.0007188
2: -0.0079258, -0.0076780, -0.0079258, -0.0076780, -0.0001698, 0.0001745
3: 0.0028884, 0.0035081, 0.0028884, 0.0035081, -0.0004247, 0.0004364
4: 0.0027627, 0.0035705, 0.0027627, 0.0035705, -0.0005535, 0.0005688
5: 0.0037417, 0.0049155, 0.0037417, 0.0049155, -0.0008044, 0.0008266
6: -0.0007183, 0.0003671, -0.0007183, 0.0003671, -0.0007644, 0.0007438
7: -0.0074047, -0.0069026, -0.0074047, -0.0069026, -0.0003536, 0.0003441
8: 0.0081031, 0.0081692, 0.0081031, 0.0081692, -0.0000466, 0.0000453
9: -0.0030091, -0.0022922, -0.0030091, -0.0022922, -0.0005049, 0.0004913

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0004715, upper bound: 0.0004646
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0004929, upper bound: 0.0004332
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0001601, 0.0006554, 0.0001601, 0.0006554, -0.0003483, 0.0003399
1: 0.9944912, 0.9955402, 0.9944912, 0.9955402, -0.0007197, 0.0007376
2: -0.0079258, -0.0076780, -0.0079258, -0.0076780, -0.0001742, 0.0001700
3: 0.0028884, 0.0035081, 0.0028884, 0.0035081, -0.0004357, 0.0004252
4: 0.0027627, 0.0035705, 0.0027627, 0.0035705, -0.0005680, 0.0005542
5: 0.0037417, 0.0049155, 0.0037417, 0.0049155, -0.0008254, 0.0008054
6: -0.0007183, 0.0003671, -0.0007183, 0.0003671, -0.0007447, 0.0007632
7: -0.0074047, -0.0069026, -0.0074047, -0.0069026, -0.0003445, 0.0003530
8: 0.0081031, 0.0081692, 0.0081031, 0.0081692, -0.0000454, 0.0000465
9: -0.0030091, -0.0022922, -0.0030091, -0.0022922, -0.0004919, 0.0005041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0004321, upper bound: 0.0004929
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0004654, upper bound: 0.0004715
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0001601, 0.0006554, 0.0001601, 0.0006554, -0.0003399, 0.0003483
1: 0.9944912, 0.9955402, 0.9944912, 0.9955402, -0.0007376, 0.0007197
2: -0.0079258, -0.0076780, -0.0079258, -0.0076780, -0.0001700, 0.0001742
3: 0.0028884, 0.0035081, 0.0028884, 0.0035081, -0.0004252, 0.0004357
4: 0.0027627, 0.0035705, 0.0027627, 0.0035705, -0.0005542, 0.0005680
5: 0.0037417, 0.0049155, 0.0037417, 0.0049155, -0.0008054, 0.0008254
6: -0.0007183, 0.0003671, -0.0007183, 0.0003671, -0.0007632, 0.0007447
7: -0.0074047, -0.0069026, -0.0074047, -0.0069026, -0.0003530, 0.0003445
8: 0.0081031, 0.0081692, 0.0081031, 0.0081692, -0.0000465, 0.0000454
9: -0.0030091, -0.0022922, -0.0030091, -0.0022922, -0.0005041, 0.0004919

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0004715, upper bound: 0.0004654
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0004929, upper bound: 0.0004321
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0001601, 0.0006554, 0.0001601, 0.0006554, -0.0003488, 0.0003394
1: 0.9944912, 0.9955402, 0.9944912, 0.9955402, -0.0007188, 0.0007387
2: -0.0079258, -0.0076780, -0.0079258, -0.0076780, -0.0001745, 0.0001698
3: 0.0028884, 0.0035081, 0.0028884, 0.0035081, -0.0004364, 0.0004247
4: 0.0027627, 0.0035705, 0.0027627, 0.0035705, -0.0005688, 0.0005535
5: 0.0037417, 0.0049155, 0.0037417, 0.0049155, -0.0008266, 0.0008044
6: -0.0007183, 0.0003671, -0.0007183, 0.0003671, -0.0007438, 0.0007644
7: -0.0074047, -0.0069026, -0.0074047, -0.0069026, -0.0003441, 0.0003536
8: 0.0081031, 0.0081692, 0.0081031, 0.0081692, -0.0000453, 0.0000466
9: -0.0030091, -0.0022922, -0.0030091, -0.0022922, -0.0004913, 0.0005049

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0004332, upper bound: 0.0004929
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0004646, upper bound: 0.0004715
time: 0.43 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.15 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.15
Output dim: 1, lower bound: -0.0004715, upper bound: 0.0004646
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.15
Output dim: 1, lower bound: -0.0004929, upper bound: 0.0004332
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.15
Output dim: 1, lower bound: -0.0004321, upper bound: 0.0004929
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.15
Output dim: 1, lower bound: -0.0004654, upper bound: 0.0004715
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.15
Output dim: 1, lower bound: -0.0004715, upper bound: 0.0004654
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.15
Output dim: 1, lower bound: -0.0004929, upper bound: 0.0004321
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.15
Output dim: 1, lower bound: -0.0004332, upper bound: 0.0004929
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.15
Output dim: 1, lower bound: -0.0004646, upper bound: 0.0004715

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0001601, 0.0006554, 0.0001601, 0.0006554, -0.0003146, 0.0003356
1: 0.9944912, 0.9955402, 0.9944912, 0.9955402, -0.0007107, 0.0006663
2: -0.0079258, -0.0076780, -0.0079258, -0.0076780, -0.0001574, 0.0001679
3: 0.0028884, 0.0035081, 0.0028884, 0.0035081, -0.0003936, 0.0004199
4: 0.0027627, 0.0035705, 0.0027627, 0.0035705, -0.0005131, 0.0005472
5: 0.0037417, 0.0049155, 0.0037417, 0.0049155, -0.0007456, 0.0007953
6: -0.0007183, 0.0003671, -0.0007183, 0.0003671, -0.0007354, 0.0006894
7: -0.0074047, -0.0069026, -0.0074047, -0.0069026, -0.0003402, 0.0003189
8: 0.0081031, 0.0081692, 0.0081031, 0.0081692, -0.0000448, 0.0000420
9: -0.0030091, -0.0022922, -0.0030091, -0.0022922, -0.0004858, 0.0004554

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 200
type: RSZ, layer: 3, pos: 208

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 3, pos: 151

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0004564, upper bound: 0.0003983
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0004555, upper bound: 0.0003942
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0001601, 0.0006554, 0.0001601, 0.0006554, -0.0003352, 0.0003151
1: 0.9944912, 0.9955402, 0.9944912, 0.9955402, -0.0006672, 0.0007098
2: -0.0079258, -0.0076780, -0.0079258, -0.0076780, -0.0001677, 0.0001576
3: 0.0028884, 0.0035081, 0.0028884, 0.0035081, -0.0004194, 0.0003942
4: 0.0027627, 0.0035705, 0.0027627, 0.0035705, -0.0005466, 0.0005138
5: 0.0037417, 0.0049155, 0.0037417, 0.0049155, -0.0007943, 0.0007466
6: -0.0007183, 0.0003671, -0.0007183, 0.0003671, -0.0006904, 0.0007345
7: -0.0074047, -0.0069026, -0.0074047, -0.0069026, -0.0003194, 0.0003398
8: 0.0081031, 0.0081692, 0.0081031, 0.0081692, -0.0000421, 0.0000447
9: -0.0030091, -0.0022922, -0.0030091, -0.0022922, -0.0004560, 0.0004852

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 200
type: RSZ, layer: 3, pos: 208

Time for candidate selection: 0.27 seconds

### Candidate
type: RSZ, layer: 3, pos: 151

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0003937, upper bound: 0.0004554
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0003962, upper bound: 0.0004564
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0001601, 0.0006554, 0.0001601, 0.0006554, -0.0003151, 0.0003352
1: 0.9944912, 0.9955402, 0.9944912, 0.9955402, -0.0007098, 0.0006672
2: -0.0079258, -0.0076780, -0.0079258, -0.0076780, -0.0001576, 0.0001677
3: 0.0028884, 0.0035081, 0.0028884, 0.0035081, -0.0003942, 0.0004194
4: 0.0027627, 0.0035705, 0.0027627, 0.0035705, -0.0005138, 0.0005466
5: 0.0037417, 0.0049155, 0.0037417, 0.0049155, -0.0007466, 0.0007943
6: -0.0007183, 0.0003671, -0.0007183, 0.0003671, -0.0007345, 0.0006904
7: -0.0074047, -0.0069026, -0.0074047, -0.0069026, -0.0003398, 0.0003194
8: 0.0081031, 0.0081692, 0.0081031, 0.0081692, -0.0000447, 0.0000421
9: -0.0030091, -0.0022922, -0.0030091, -0.0022922, -0.0004852, 0.0004560

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 200
type: RSZ, layer: 3, pos: 208

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 3, pos: 151

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0004564, upper bound: 0.0003962
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0004554, upper bound: 0.0003937
time: 0.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0001601, 0.0006554, 0.0001601, 0.0006554, -0.0003356, 0.0003146
1: 0.9944912, 0.9955402, 0.9944912, 0.9955402, -0.0006663, 0.0007107
2: -0.0079258, -0.0076780, -0.0079258, -0.0076780, -0.0001679, 0.0001574
3: 0.0028884, 0.0035081, 0.0028884, 0.0035081, -0.0004199, 0.0003936
4: 0.0027627, 0.0035705, 0.0027627, 0.0035705, -0.0005472, 0.0005131
5: 0.0037417, 0.0049155, 0.0037417, 0.0049155, -0.0007953, 0.0007456
6: -0.0007183, 0.0003671, -0.0007183, 0.0003671, -0.0006894, 0.0007354
7: -0.0074047, -0.0069026, -0.0074047, -0.0069026, -0.0003189, 0.0003402
8: 0.0081031, 0.0081692, 0.0081031, 0.0081692, -0.0000420, 0.0000448
9: -0.0030091, -0.0022922, -0.0030091, -0.0022922, -0.0004554, 0.0004858

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 200
type: RSZ, layer: 3, pos: 208

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 3, pos: 151

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0003942, upper bound: 0.0004555
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0003983, upper bound: 0.0004564
time: 0.45 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.28 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.28
Output dim: 1, lower bound: -0.0004564, upper bound: 0.0003983
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.28
Output dim: 1, lower bound: -0.0004555, upper bound: 0.0003942
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.28
Output dim: 1, lower bound: -0.0003937, upper bound: 0.0004554
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.28
Output dim: 1, lower bound: -0.0003962, upper bound: 0.0004564
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.28
Output dim: 1, lower bound: -0.0004564, upper bound: 0.0003962
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.28
Output dim: 1, lower bound: -0.0004554, upper bound: 0.0003937
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.28
Output dim: 1, lower bound: -0.0003942, upper bound: 0.0004555
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.28
Output dim: 1, lower bound: -0.0003983, upper bound: 0.0004564

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.52 + 23.41 = 25.93 seconds
