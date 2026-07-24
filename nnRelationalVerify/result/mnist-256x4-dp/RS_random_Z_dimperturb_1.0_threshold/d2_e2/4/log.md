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
execution time: IAR + RelationalAnalysis = 1.27 + 2.25 = 3.52 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0495323, upper bound: 0.0495323

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 201

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0488680, upper bound: 0.0488669
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0488669, upper bound: 0.0488680
time: 1.40 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.84 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.84
Output dim: 0, lower bound: -0.0488680, upper bound: 0.0488669
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.84
Output dim: 0, lower bound: -0.0488669, upper bound: 0.0488680

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
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0488036, upper bound: 0.0488014
time: 1.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0488023, upper bound: 0.0488026
time: 1.45 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0483247, upper bound: 0.0482656
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0482655, upper bound: 0.0483247
time: 1.55 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.13 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.13
Output dim: 0, lower bound: -0.0488036, upper bound: 0.0488014
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.13
Output dim: 0, lower bound: -0.0488023, upper bound: 0.0488026
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.13
Output dim: 0, lower bound: -0.0483247, upper bound: 0.0482656
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.13
Output dim: 0, lower bound: -0.0482655, upper bound: 0.0483247

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
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0487863, upper bound: 0.0487242
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0487242, upper bound: 0.0487839
time: 1.68 seconds

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
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0486317, upper bound: 0.0484986
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0484978, upper bound: 0.0486312
time: 1.55 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0475232, upper bound: 0.0476271
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0476875, upper bound: 0.0474756
time: 1.31 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0464558, upper bound: 0.0465140
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0464558, upper bound: 0.0465140
time: 1.31 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.96 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.96
Output dim: 0, lower bound: -0.0487863, upper bound: 0.0487242
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.96
Output dim: 0, lower bound: -0.0487242, upper bound: 0.0487839
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.96
Output dim: 0, lower bound: -0.0486317, upper bound: 0.0484986
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.96
Output dim: 0, lower bound: -0.0484978, upper bound: 0.0486312
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.96
Output dim: 0, lower bound: -0.0475232, upper bound: 0.0476271
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.96
Output dim: 0, lower bound: -0.0476875, upper bound: 0.0474756
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.96
Output dim: 0, lower bound: -0.0464558, upper bound: 0.0465140
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.96
Output dim: 0, lower bound: -0.0464558, upper bound: 0.0465140

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 201

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0464372, upper bound: 0.0463899
time: 2.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0464372, upper bound: 0.0463899
time: 2.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0485401, upper bound: 0.0484796
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0484541, upper bound: 0.0486138
time: 1.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0480912, upper bound: 0.0479036
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0480313, upper bound: 0.0479566
time: 1.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0477259, upper bound: 0.0480227
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0478909, upper bound: 0.0478491
time: 1.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0456722, upper bound: 0.0457810
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0456722, upper bound: 0.0457810
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0476053, upper bound: 0.0473643
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0475828, upper bound: 0.0473949
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 201

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0435543, upper bound: 0.0436138
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0435543, upper bound: 0.0436138
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450499, upper bound: 0.0451175
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450499, upper bound: 0.0451175
time: 1.50 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.84 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 0, lower bound: -0.0464372, upper bound: 0.0463899
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 0, lower bound: -0.0464372, upper bound: 0.0463899
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 0, lower bound: -0.0485401, upper bound: 0.0484796
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 0, lower bound: -0.0484541, upper bound: 0.0486138
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 0, lower bound: -0.0480912, upper bound: 0.0479036
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 0, lower bound: -0.0480313, upper bound: 0.0479566
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 0, lower bound: -0.0477259, upper bound: 0.0480227
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 0, lower bound: -0.0478909, upper bound: 0.0478491
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.84
Output dim: 0, lower bound: -0.0456722, upper bound: 0.0457810
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.84
Output dim: 0, lower bound: -0.0456722, upper bound: 0.0457810
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 0, lower bound: -0.0476053, upper bound: 0.0473643
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 0, lower bound: -0.0475828, upper bound: 0.0473949
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.84
Output dim: 0, lower bound: -0.0435543, upper bound: 0.0436138
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.84
Output dim: 0, lower bound: -0.0435543, upper bound: 0.0436138
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.84
Output dim: 0, lower bound: -0.0450499, upper bound: 0.0451175
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.84
Output dim: 0, lower bound: -0.0450499, upper bound: 0.0451175

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0463578, upper bound: 0.0463033
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0463438, upper bound: 0.0463081
time: 1.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462440, upper bound: 0.0461031
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0461233, upper bound: 0.0461880
time: 1.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 201

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0453026, upper bound: 0.0452480
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0453026, upper bound: 0.0452480
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0483745, upper bound: 0.0485139
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0483594, upper bound: 0.0485366
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 201

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0459197, upper bound: 0.0457313
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0459197, upper bound: 0.0457313
time: 1.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0464732, upper bound: 0.0463950
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0464732, upper bound: 0.0463950
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0477077, upper bound: 0.0479348
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0476774, upper bound: 0.0480057
time: 1.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0478132, upper bound: 0.0477433
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0477912, upper bound: 0.0477712
time: 1.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0474354, upper bound: 0.0470818
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0472978, upper bound: 0.0471868
time: 1.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0437982, upper bound: 0.0436557
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0437982, upper bound: 0.0436557
time: 1.48 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.16 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -0.0463578, upper bound: 0.0463033
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -0.0463438, upper bound: 0.0463081
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -0.0462440, upper bound: 0.0461031
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -0.0461233, upper bound: 0.0461880
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 0, lower bound: -0.0453026, upper bound: 0.0452480
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 0, lower bound: -0.0453026, upper bound: 0.0452480
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -0.0483745, upper bound: 0.0485139
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -0.0483594, upper bound: 0.0485366
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -0.0459197, upper bound: 0.0457313
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -0.0459197, upper bound: 0.0457313
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -0.0464732, upper bound: 0.0463950
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -0.0464732, upper bound: 0.0463950
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -0.0477077, upper bound: 0.0479348
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -0.0476774, upper bound: 0.0480057
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -0.0478132, upper bound: 0.0477433
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -0.0477912, upper bound: 0.0477712
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -0.0474354, upper bound: 0.0470818
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -0.0472978, upper bound: 0.0471868
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 0, lower bound: -0.0437982, upper bound: 0.0436557
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 0, lower bound: -0.0437982, upper bound: 0.0436557

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0461647, upper bound: 0.0460122
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0460442, upper bound: 0.0461016
time: 1.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0421745, upper bound: 0.0421483
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0421745, upper bound: 0.0421483
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0459035, upper bound: 0.0456894
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0458342, upper bound: 0.0457635
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0418038, upper bound: 0.0417880
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0418038, upper bound: 0.0417880
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0460225, upper bound: 0.0461504
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0460225, upper bound: 0.0461504
time: 1.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 201

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0478117, upper bound: 0.0479337
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0477377, upper bound: 0.0479909
time: 1.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0412251, upper bound: 0.0411949
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0412251, upper bound: 0.0411949
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0447246, upper bound: 0.0445156
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0447246, upper bound: 0.0445156
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0463963, upper bound: 0.0463002
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0463717, upper bound: 0.0463146
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0463963, upper bound: 0.0463002
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0463717, upper bound: 0.0463146
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0472612, upper bound: 0.0474509
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0472178, upper bound: 0.0474616
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0472292, upper bound: 0.0475186
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0471909, upper bound: 0.0475393
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0458288, upper bound: 0.0457995
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0458288, upper bound: 0.0457995
time: 1.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0441188, upper bound: 0.0441779
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0441188, upper bound: 0.0441780
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0427507, upper bound: 0.0426111
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0427507, upper bound: 0.0426111
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0457879, upper bound: 0.0456598
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0457879, upper bound: 0.0456598
time: 1.35 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.72 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -0.0461647, upper bound: 0.0460122
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -0.0460442, upper bound: 0.0461016
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.72
Output dim: 0, lower bound: -0.0421745, upper bound: 0.0421483
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.72
Output dim: 0, lower bound: -0.0421745, upper bound: 0.0421483
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -0.0459035, upper bound: 0.0456894
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -0.0458342, upper bound: 0.0457635
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.72
Output dim: 0, lower bound: -0.0418038, upper bound: 0.0417880
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.72
Output dim: 0, lower bound: -0.0418038, upper bound: 0.0417880
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -0.0460225, upper bound: 0.0461504
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -0.0460225, upper bound: 0.0461504
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -0.0478117, upper bound: 0.0479337
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -0.0477377, upper bound: 0.0479909
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.72
Output dim: 0, lower bound: -0.0412251, upper bound: 0.0411949
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.72
Output dim: 0, lower bound: -0.0412251, upper bound: 0.0411949
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.72
Output dim: 0, lower bound: -0.0447246, upper bound: 0.0445156
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.72
Output dim: 0, lower bound: -0.0447246, upper bound: 0.0445156
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -0.0463963, upper bound: 0.0463002
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -0.0463717, upper bound: 0.0463146
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -0.0463963, upper bound: 0.0463002
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -0.0463717, upper bound: 0.0463146
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -0.0472612, upper bound: 0.0474509
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -0.0472178, upper bound: 0.0474616
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -0.0472292, upper bound: 0.0475186
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -0.0471909, upper bound: 0.0475393
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -0.0458288, upper bound: 0.0457995
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -0.0458288, upper bound: 0.0457995
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.72
Output dim: 0, lower bound: -0.0441188, upper bound: 0.0441779
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.72
Output dim: 0, lower bound: -0.0441188, upper bound: 0.0441780
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.72
Output dim: 0, lower bound: -0.0427507, upper bound: 0.0426111
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.72
Output dim: 0, lower bound: -0.0427507, upper bound: 0.0426111
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.72
Output dim: 0, lower bound: -0.0457879, upper bound: 0.0456598
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.72
Output dim: 0, lower bound: -0.0457879, upper bound: 0.0456598

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 201

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0458220, upper bound: 0.0455876
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0457544, upper bound: 0.0456713
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451993, upper bound: 0.0453694
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0453084, upper bound: 0.0452295
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 201

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0436505, upper bound: 0.0434685
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0436505, upper bound: 0.0434685
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0453072, upper bound: 0.0452215
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0452850, upper bound: 0.0452423
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0417117, upper bound: 0.0417251
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0417117, upper bound: 0.0417251
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 201

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0418458, upper bound: 0.0419921
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0418458, upper bound: 0.0419921
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0459992, upper bound: 0.0461402
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0459992, upper bound: 0.0461402
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0459341, upper bound: 0.0461950
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0459341, upper bound: 0.0461950
time: 1.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0429185, upper bound: 0.0427770
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0429174, upper bound: 0.0427770
time: 1.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0447355, upper bound: 0.0446698
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0447355, upper bound: 0.0446698
time: 1.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 201

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0441871, upper bound: 0.0441184
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0441871, upper bound: 0.0441184
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0458764, upper bound: 0.0457989
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0458475, upper bound: 0.0458234
time: 1.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0436616, upper bound: 0.0438439
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0436616, upper bound: 0.0438439
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0446655, upper bound: 0.0448729
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0446655, upper bound: 0.0448729
time: 1.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0471483, upper bound: 0.0474239
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0471328, upper bound: 0.0474412
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0466092, upper bound: 0.0469109
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0465638, upper bound: 0.0469570
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0458118, upper bound: 0.0457221
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0457925, upper bound: 0.0457838
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0441425, upper bound: 0.0441272
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0441425, upper bound: 0.0441272
time: 1.46 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 4.39 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 0, lower bound: -0.0458220, upper bound: 0.0455876
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 0, lower bound: -0.0457544, upper bound: 0.0456713
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 0, lower bound: -0.0451993, upper bound: 0.0453694
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 0, lower bound: -0.0453084, upper bound: 0.0452295
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 0, lower bound: -0.0436505, upper bound: 0.0434685
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 0, lower bound: -0.0436505, upper bound: 0.0434685
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 0, lower bound: -0.0453072, upper bound: 0.0452215
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 0, lower bound: -0.0452850, upper bound: 0.0452423
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 0, lower bound: -0.0417117, upper bound: 0.0417251
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 0, lower bound: -0.0417117, upper bound: 0.0417251
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 0, lower bound: -0.0418458, upper bound: 0.0419921
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 0, lower bound: -0.0418458, upper bound: 0.0419921
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 0, lower bound: -0.0459992, upper bound: 0.0461402
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 0, lower bound: -0.0459992, upper bound: 0.0461402
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 0, lower bound: -0.0459341, upper bound: 0.0461950
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 0, lower bound: -0.0459341, upper bound: 0.0461950
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 0, lower bound: -0.0429185, upper bound: 0.0427770
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 0, lower bound: -0.0429174, upper bound: 0.0427770
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 0, lower bound: -0.0447355, upper bound: 0.0446698
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 0, lower bound: -0.0447355, upper bound: 0.0446698
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 0, lower bound: -0.0441871, upper bound: 0.0441184
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 0, lower bound: -0.0441871, upper bound: 0.0441184
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 0, lower bound: -0.0458764, upper bound: 0.0457989
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 0, lower bound: -0.0458475, upper bound: 0.0458234
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 0, lower bound: -0.0436616, upper bound: 0.0438439
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 0, lower bound: -0.0436616, upper bound: 0.0438439
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 0, lower bound: -0.0446655, upper bound: 0.0448729
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 0, lower bound: -0.0446655, upper bound: 0.0448729
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 0, lower bound: -0.0471483, upper bound: 0.0474239
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 0, lower bound: -0.0471328, upper bound: 0.0474412
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 0, lower bound: -0.0466092, upper bound: 0.0469109
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 0, lower bound: -0.0465638, upper bound: 0.0469570
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 0, lower bound: -0.0458118, upper bound: 0.0457221
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 0, lower bound: -0.0457925, upper bound: 0.0457838
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 0, lower bound: -0.0441425, upper bound: 0.0441272
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 0, lower bound: -0.0441425, upper bound: 0.0441272

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0411349, upper bound: 0.0410443
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0411349, upper bound: 0.0410443
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451626, upper bound: 0.0454681
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0453419, upper bound: 0.0453064
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0444567, upper bound: 0.0445648
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0444567, upper bound: 0.0445648
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0454743, upper bound: 0.0456828
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0454505, upper bound: 0.0456913
time: 1.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0423915, upper bound: 0.0426085
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0423915, upper bound: 0.0426085
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0436545, upper bound: 0.0435902
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0436545, upper bound: 0.0435902
time: 1.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 201

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0425620, upper bound: 0.0425446
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0425620, upper bound: 0.0425446
time: 1.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451350, upper bound: 0.0454220
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451350, upper bound: 0.0454220
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0445994, upper bound: 0.0448314
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0445994, upper bound: 0.0448314
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0465244, upper bound: 0.0467952
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0465067, upper bound: 0.0468304
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0428300, upper bound: 0.0431802
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0428300, upper bound: 0.0431802
time: 1.06 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.33 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -0.0411349, upper bound: 0.0410443
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -0.0411349, upper bound: 0.0410443
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -0.0451626, upper bound: 0.0454681
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -0.0453419, upper bound: 0.0453064
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -0.0444567, upper bound: 0.0445648
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -0.0444567, upper bound: 0.0445648
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -0.0454743, upper bound: 0.0456828
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -0.0454505, upper bound: 0.0456913
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -0.0423915, upper bound: 0.0426085
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -0.0423915, upper bound: 0.0426085
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -0.0436545, upper bound: 0.0435902
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -0.0436545, upper bound: 0.0435902
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -0.0425620, upper bound: 0.0425446
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -0.0425620, upper bound: 0.0425446
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -0.0451350, upper bound: 0.0454220
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -0.0451350, upper bound: 0.0454220
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -0.0445994, upper bound: 0.0448314
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -0.0445994, upper bound: 0.0448314
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -0.0465244, upper bound: 0.0467952
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -0.0465067, upper bound: 0.0468304
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -0.0428300, upper bound: 0.0431802
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -0.0428300, upper bound: 0.0431802

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 201

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0430540, upper bound: 0.0432495
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0430540, upper bound: 0.0432495
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449842, upper bound: 0.0453206
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449842, upper bound: 0.0453206
time: 1.36 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 3.92 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.92
Output dim: 0, lower bound: -0.0430540, upper bound: 0.0432495
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.92
Output dim: 0, lower bound: -0.0430540, upper bound: 0.0432495
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.92
Output dim: 0, lower bound: -0.0449842, upper bound: 0.0453206
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.92
Output dim: 0, lower bound: -0.0449842, upper bound: 0.0453206

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.52 + 275.33 = 278.85 seconds
