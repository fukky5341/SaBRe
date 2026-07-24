## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.045817377500000006


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9380678, 0.9917637, 0.9380678, 0.9917637, -0.0536959, 0.0536959)
1: (-0.0115945, -0.0026022, -0.0115945, -0.0026022, -0.0089923, 0.0089923)
2: (0.0075203, 0.0204642, 0.0075203, 0.0204642, -0.0129439, 0.0129439)
3: (-0.0090238, 0.0055741, -0.0090238, 0.0055741, -0.0145979, 0.0145979)
4: (-0.0041402, 0.0057734, -0.0041402, 0.0057734, -0.0099136, 0.0099136)
5: (0.0084180, 0.0549622, 0.0084180, 0.0549622, -0.0465442, 0.0465442)
6: (-0.0082169, 0.0051646, -0.0082169, 0.0051646, -0.0133815, 0.0133815)
7: (-0.0184287, -0.0037830, -0.0184287, -0.0037830, -0.0146457, 0.0146457)
8: (-0.0084818, 0.0135686, -0.0084818, 0.0135686, -0.0220504, 0.0220504)
9: (-0.0001674, 0.0124900, -0.0001674, 0.0124900, -0.0126574, 0.0126574)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.26 + 2.27 = 3.52 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0495323, upper bound: 0.0495323

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0494671, upper bound: 0.0494634
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0494634, upper bound: 0.0494671
time: 1.37 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 3.09 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 3.09
Output dim: 0, lower bound: -0.0494671, upper bound: 0.0494634
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 3.09
Output dim: 0, lower bound: -0.0494634, upper bound: 0.0494671

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.9380678, 0.9917637, 0.9380678, 0.9917637, -0.0536959, 0.0536959
1: -0.0115945, -0.0026022, -0.0115945, -0.0026022, -0.0089923, 0.0089923
2: 0.0075203, 0.0204642, 0.0075203, 0.0204642, -0.0129439, 0.0129439
3: -0.0090238, 0.0055741, -0.0090238, 0.0055741, -0.0145979, 0.0145979
4: -0.0041402, 0.0057734, -0.0041402, 0.0057734, -0.0099136, 0.0099136
5: 0.0084180, 0.0549622, 0.0084180, 0.0549622, -0.0465442, 0.0465442
6: -0.0082169, 0.0051646, -0.0082169, 0.0051646, -0.0133815, 0.0133815
7: -0.0184287, -0.0037830, -0.0184287, -0.0037830, -0.0146457, 0.0146457
8: -0.0084818, 0.0135686, -0.0084818, 0.0135686, -0.0220504, 0.0220504
9: -0.0001674, 0.0124900, -0.0001674, 0.0124900, -0.0126574, 0.0126574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0492460, upper bound: 0.0491183
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0491203, upper bound: 0.0492452
time: 1.35 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.9380678, 0.9917637, 0.9380678, 0.9917637, -0.0536959, 0.0536959
1: -0.0115945, -0.0026022, -0.0115945, -0.0026022, -0.0089923, 0.0089923
2: 0.0075203, 0.0204642, 0.0075203, 0.0204642, -0.0129439, 0.0129439
3: -0.0090238, 0.0055741, -0.0090238, 0.0055741, -0.0145979, 0.0145979
4: -0.0041402, 0.0057734, -0.0041402, 0.0057734, -0.0099136, 0.0099136
5: 0.0084180, 0.0549622, 0.0084180, 0.0549622, -0.0465442, 0.0465442
6: -0.0082169, 0.0051646, -0.0082169, 0.0051646, -0.0133815, 0.0133815
7: -0.0184287, -0.0037830, -0.0184287, -0.0037830, -0.0146457, 0.0146457
8: -0.0084818, 0.0135686, -0.0084818, 0.0135686, -0.0220504, 0.0220504
9: -0.0001674, 0.0124900, -0.0001674, 0.0124900, -0.0126574, 0.0126574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0492452, upper bound: 0.0491203
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0491183, upper bound: 0.0492460
time: 1.71 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.70 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.70
Output dim: 0, lower bound: -0.0492460, upper bound: 0.0491183
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.70
Output dim: 0, lower bound: -0.0491203, upper bound: 0.0492452
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.70
Output dim: 0, lower bound: -0.0492452, upper bound: 0.0491203
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.70
Output dim: 0, lower bound: -0.0491183, upper bound: 0.0492460

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9380678, 0.9917637, 0.9380678, 0.9917637, -0.0536959, 0.0536959
1: -0.0115945, -0.0026022, -0.0115945, -0.0026022, -0.0089923, 0.0089923
2: 0.0075203, 0.0204642, 0.0075203, 0.0204642, -0.0129439, 0.0129439
3: -0.0090238, 0.0055741, -0.0090238, 0.0055741, -0.0145979, 0.0145979
4: -0.0041402, 0.0057734, -0.0041402, 0.0057734, -0.0099136, 0.0099136
5: 0.0084180, 0.0549622, 0.0084180, 0.0549622, -0.0465442, 0.0465442
6: -0.0082169, 0.0051646, -0.0082169, 0.0051646, -0.0133815, 0.0133815
7: -0.0184287, -0.0037830, -0.0184287, -0.0037830, -0.0146457, 0.0146457
8: -0.0084818, 0.0135686, -0.0084818, 0.0135686, -0.0220504, 0.0220504
9: -0.0001674, 0.0124900, -0.0001674, 0.0124900, -0.0126574, 0.0126574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0456656, upper bound: 0.0454961
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0456656, upper bound: 0.0454961
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9380678, 0.9917637, 0.9380678, 0.9917637, -0.0536959, 0.0536959
1: -0.0115945, -0.0026022, -0.0115945, -0.0026022, -0.0089923, 0.0089923
2: 0.0075203, 0.0204642, 0.0075203, 0.0204642, -0.0129439, 0.0129439
3: -0.0090238, 0.0055741, -0.0090238, 0.0055741, -0.0145979, 0.0145979
4: -0.0041402, 0.0057734, -0.0041402, 0.0057734, -0.0099136, 0.0099136
5: 0.0084180, 0.0549622, 0.0084180, 0.0549622, -0.0465442, 0.0465442
6: -0.0082169, 0.0051646, -0.0082169, 0.0051646, -0.0133815, 0.0133815
7: -0.0184287, -0.0037830, -0.0184287, -0.0037830, -0.0146457, 0.0146457
8: -0.0084818, 0.0135686, -0.0084818, 0.0135686, -0.0220504, 0.0220504
9: -0.0001674, 0.0124900, -0.0001674, 0.0124900, -0.0126574, 0.0126574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0455021, upper bound: 0.0456655
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0455021, upper bound: 0.0456655
time: 1.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9380678, 0.9917637, 0.9380678, 0.9917637, -0.0536959, 0.0536959
1: -0.0115945, -0.0026022, -0.0115945, -0.0026022, -0.0089923, 0.0089923
2: 0.0075203, 0.0204642, 0.0075203, 0.0204642, -0.0129439, 0.0129439
3: -0.0090238, 0.0055741, -0.0090238, 0.0055741, -0.0145979, 0.0145979
4: -0.0041402, 0.0057734, -0.0041402, 0.0057734, -0.0099136, 0.0099136
5: 0.0084180, 0.0549622, 0.0084180, 0.0549622, -0.0465442, 0.0465442
6: -0.0082169, 0.0051646, -0.0082169, 0.0051646, -0.0133815, 0.0133815
7: -0.0184287, -0.0037830, -0.0184287, -0.0037830, -0.0146457, 0.0146457
8: -0.0084818, 0.0135686, -0.0084818, 0.0135686, -0.0220504, 0.0220504
9: -0.0001674, 0.0124900, -0.0001674, 0.0124900, -0.0126574, 0.0126574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0456655, upper bound: 0.0455021
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0456655, upper bound: 0.0455021
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9380678, 0.9917637, 0.9380678, 0.9917637, -0.0536959, 0.0536959
1: -0.0115945, -0.0026022, -0.0115945, -0.0026022, -0.0089923, 0.0089923
2: 0.0075203, 0.0204642, 0.0075203, 0.0204642, -0.0129439, 0.0129439
3: -0.0090238, 0.0055741, -0.0090238, 0.0055741, -0.0145979, 0.0145979
4: -0.0041402, 0.0057734, -0.0041402, 0.0057734, -0.0099136, 0.0099136
5: 0.0084180, 0.0549622, 0.0084180, 0.0549622, -0.0465442, 0.0465442
6: -0.0082169, 0.0051646, -0.0082169, 0.0051646, -0.0133815, 0.0133815
7: -0.0184287, -0.0037830, -0.0184287, -0.0037830, -0.0146457, 0.0146457
8: -0.0084818, 0.0135686, -0.0084818, 0.0135686, -0.0220504, 0.0220504
9: -0.0001674, 0.0124900, -0.0001674, 0.0124900, -0.0126574, 0.0126574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0454961, upper bound: 0.0456656
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0454961, upper bound: 0.0456656
time: 1.33 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.97 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.97
Output dim: 0, lower bound: -0.0456656, upper bound: 0.0454961
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.97
Output dim: 0, lower bound: -0.0456656, upper bound: 0.0454961
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.97
Output dim: 0, lower bound: -0.0455021, upper bound: 0.0456655
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.97
Output dim: 0, lower bound: -0.0455021, upper bound: 0.0456655
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.97
Output dim: 0, lower bound: -0.0456655, upper bound: 0.0455021
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.97
Output dim: 0, lower bound: -0.0456655, upper bound: 0.0455021
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.97
Output dim: 0, lower bound: -0.0454961, upper bound: 0.0456656
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.97
Output dim: 0, lower bound: -0.0454961, upper bound: 0.0456656

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.52 + 27.67 = 31.19 seconds
