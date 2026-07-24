## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 5.355e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0037710, -0.0031327, -0.0037710, -0.0031327, -0.0006383, 0.0006383)
1: (0.0059706, 0.0063943, 0.0059706, 0.0063943, -0.0004237, 0.0004237)
2: (0.0110240, 0.0122354, 0.0110240, 0.0122354, -0.0009910, 0.0009910)
3: (-0.0035409, -0.0030237, -0.0035409, -0.0030237, -0.0005172, 0.0005172)
4: (0.0049489, 0.0051230, 0.0049489, 0.0051230, -0.0000969, 0.0000969)
5: (-0.0014843, -0.0010727, -0.0014843, -0.0010727, -0.0004116, 0.0004116)
6: (-0.0055979, -0.0054069, -0.0055979, -0.0054069, -0.0001910, 0.0001910)
7: (-0.0030353, -0.0027112, -0.0030353, -0.0027112, -0.0003242, 0.0003242)
8: (-0.0025715, -0.0017671, -0.0025715, -0.0017671, -0.0008044, 0.0008044)
9: (1.0004611, 1.0005401, 1.0004611, 1.0005401, -0.0000790, 0.0000790)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.19 + 1.29 = 2.47 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0000754, upper bound: 0.0000754

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0000686, upper bound: 0.0000686
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0000686, upper bound: 0.0000686
time: 0.43 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.88 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.88
Output dim: 9, lower bound: -0.0000686, upper bound: 0.0000686
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.88
Output dim: 9, lower bound: -0.0000686, upper bound: 0.0000686

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037710, -0.0031327, -0.0037710, -0.0031327, -0.0006383, 0.0006383
1: 0.0059706, 0.0063943, 0.0059706, 0.0063943, -0.0004237, 0.0004237
2: 0.0110240, 0.0122354, 0.0110240, 0.0122354, -0.0009907, 0.0009912
3: -0.0035409, -0.0030237, -0.0035409, -0.0030237, -0.0005172, 0.0005172
4: 0.0049489, 0.0051230, 0.0049489, 0.0051230, -0.0000968, 0.0000967
5: -0.0014843, -0.0010727, -0.0014843, -0.0010727, -0.0004116, 0.0004116
6: -0.0055979, -0.0054069, -0.0055979, -0.0054069, -0.0001910, 0.0001910
7: -0.0030353, -0.0027112, -0.0030353, -0.0027112, -0.0003242, 0.0003242
8: -0.0025715, -0.0017671, -0.0025715, -0.0017671, -0.0008044, 0.0008044
9: 1.0004611, 1.0005401, 1.0004611, 1.0005401, -0.0000790, 0.0000790

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0000485, upper bound: 0.0000485
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0000485, upper bound: 0.0000485
time: 0.38 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037710, -0.0031327, -0.0037710, -0.0031327, -0.0006383, 0.0006383
1: 0.0059706, 0.0063943, 0.0059706, 0.0063943, -0.0004237, 0.0004237
2: 0.0110240, 0.0122354, 0.0110240, 0.0122354, -0.0009910, 0.0009907
3: -0.0035409, -0.0030237, -0.0035409, -0.0030237, -0.0005172, 0.0005172
4: 0.0049489, 0.0051230, 0.0049489, 0.0051230, -0.0000969, 0.0000968
5: -0.0014843, -0.0010727, -0.0014843, -0.0010727, -0.0004116, 0.0004116
6: -0.0055979, -0.0054069, -0.0055979, -0.0054069, -0.0001910, 0.0001910
7: -0.0030353, -0.0027112, -0.0030353, -0.0027112, -0.0003242, 0.0003242
8: -0.0025715, -0.0017671, -0.0025715, -0.0017671, -0.0008044, 0.0008044
9: 1.0004611, 1.0005401, 1.0004611, 1.0005401, -0.0000790, 0.0000790

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0000685, upper bound: 0.0000685
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0000685, upper bound: 0.0000685
time: 0.43 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 1.93 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 1.93
Output dim: 9, lower bound: -0.0000485, upper bound: 0.0000485
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 1.93
Output dim: 9, lower bound: -0.0000485, upper bound: 0.0000485
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 1.93
Output dim: 9, lower bound: -0.0000685, upper bound: 0.0000685
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 1.93
Output dim: 9, lower bound: -0.0000685, upper bound: 0.0000685

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037710, -0.0031327, -0.0037710, -0.0031327, -0.0006383, 0.0006383
1: 0.0059706, 0.0063943, 0.0059706, 0.0063943, -0.0004237, 0.0004237
2: 0.0110240, 0.0122354, 0.0110240, 0.0122354, -0.0009729, 0.0009729
3: -0.0035409, -0.0030237, -0.0035409, -0.0030237, -0.0005172, 0.0005172
4: 0.0049489, 0.0051230, 0.0049489, 0.0051230, -0.0000828, 0.0000845
5: -0.0014843, -0.0010727, -0.0014843, -0.0010727, -0.0004116, 0.0004116
6: -0.0055979, -0.0054069, -0.0055979, -0.0054069, -0.0001910, 0.0001910
7: -0.0030353, -0.0027112, -0.0030353, -0.0027112, -0.0003242, 0.0003242
8: -0.0025715, -0.0017671, -0.0025715, -0.0017671, -0.0008044, 0.0008044
9: 1.0004611, 1.0005401, 1.0004611, 1.0005401, -0.0000790, 0.0000790

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0000465, upper bound: 0.0000465
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0000465, upper bound: 0.0000465
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037710, -0.0031327, -0.0037710, -0.0031327, -0.0006383, 0.0006383
1: 0.0059706, 0.0063943, 0.0059706, 0.0063943, -0.0004237, 0.0004237
2: 0.0110240, 0.0122354, 0.0110240, 0.0122354, -0.0009732, 0.0009907
3: -0.0035409, -0.0030237, -0.0035409, -0.0030237, -0.0005172, 0.0005172
4: 0.0049489, 0.0051230, 0.0049489, 0.0051230, -0.0000969, 0.0000827
5: -0.0014843, -0.0010727, -0.0014843, -0.0010727, -0.0004116, 0.0004116
6: -0.0055979, -0.0054069, -0.0055979, -0.0054069, -0.0001910, 0.0001910
7: -0.0030353, -0.0027112, -0.0030353, -0.0027112, -0.0003242, 0.0003242
8: -0.0025715, -0.0017671, -0.0025715, -0.0017671, -0.0008044, 0.0008044
9: 1.0004611, 1.0005401, 1.0004611, 1.0005401, -0.0000790, 0.0000790

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0000465, upper bound: 0.0000465
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0000465, upper bound: 0.0000465
time: 0.39 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 1.93 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 1.93
Output dim: 9, lower bound: -0.0000465, upper bound: 0.0000465
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 1.93
Output dim: 9, lower bound: -0.0000465, upper bound: 0.0000465
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 1.93
Output dim: 9, lower bound: -0.0000465, upper bound: 0.0000465
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 1.93
Output dim: 9, lower bound: -0.0000465, upper bound: 0.0000465

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.47 + 8.50 = 10.97 seconds
