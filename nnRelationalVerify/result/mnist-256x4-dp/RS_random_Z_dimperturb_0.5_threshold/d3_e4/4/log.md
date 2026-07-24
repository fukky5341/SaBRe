## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00913976


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724)
1: (-0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842)
2: (-0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222)
3: (-0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994)
4: (-0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739)
5: (-0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787)
6: (-0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309)
7: (-0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951)
8: (0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211)
9: (-0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0188877, 0.0188877)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.69 + 2.37 = 4.06 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0126713, upper bound: 0.0126708

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125405, upper bound: 0.0125701
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125707, upper bound: 0.0125399
time: 1.41 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.92 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.92
Output dim: 8, lower bound: -0.0125405, upper bound: 0.0125701
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.92
Output dim: 8, lower bound: -0.0125707, upper bound: 0.0125399

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0188877, 0.0188877

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124485, upper bound: 0.0124760
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124486, upper bound: 0.0124758
time: 1.29 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0188877, 0.0188877

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123847, upper bound: 0.0122890
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123365, upper bound: 0.0123583
time: 2.04 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.90 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.90
Output dim: 8, lower bound: -0.0124485, upper bound: 0.0124760
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.90
Output dim: 8, lower bound: -0.0124486, upper bound: 0.0124758
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.90
Output dim: 8, lower bound: -0.0123847, upper bound: 0.0122890
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.90
Output dim: 8, lower bound: -0.0123365, upper bound: 0.0123583

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0188877, 0.0188877

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116844, upper bound: 0.0116926
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116844, upper bound: 0.0116921
time: 2.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0188877, 0.0188877

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123454, upper bound: 0.0123860
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123543, upper bound: 0.0123728
time: 1.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0188877, 0.0188877

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119765, upper bound: 0.0120830
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121724, upper bound: 0.0118579
time: 1.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0188877, 0.0188877

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118504, upper bound: 0.0118398
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118504, upper bound: 0.0118398
time: 1.25 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.05 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.05
Output dim: 8, lower bound: -0.0116844, upper bound: 0.0116926
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.05
Output dim: 8, lower bound: -0.0116844, upper bound: 0.0116921
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.05
Output dim: 8, lower bound: -0.0123454, upper bound: 0.0123860
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.05
Output dim: 8, lower bound: -0.0123543, upper bound: 0.0123728
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.05
Output dim: 8, lower bound: -0.0119765, upper bound: 0.0120830
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.05
Output dim: 8, lower bound: -0.0121724, upper bound: 0.0118579
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.05
Output dim: 8, lower bound: -0.0118504, upper bound: 0.0118398
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.05
Output dim: 8, lower bound: -0.0118504, upper bound: 0.0118398

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0188877, 0.0188877

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115195, upper bound: 0.0115636
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115564, upper bound: 0.0115331
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0188877, 0.0188877

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112625, upper bound: 0.0113264
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113178, upper bound: 0.0112659
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0188877, 0.0188877

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121347, upper bound: 0.0121383
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120790, upper bound: 0.0121778
time: 2.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0188877, 0.0188877

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120910, upper bound: 0.0120999
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120910, upper bound: 0.0121978
time: 1.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0179375, 0.0179258

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115632, upper bound: 0.0116713
time: 1.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115632, upper bound: 0.0116713
time: 2.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0179175, 0.0179454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118989, upper bound: 0.0116182
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119300, upper bound: 0.0115815
time: 1.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0188877, 0.0188877

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114565, upper bound: 0.0115863
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115915, upper bound: 0.0114481
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0188877, 0.0188877

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115376, upper bound: 0.0115638
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115748, upper bound: 0.0115283
time: 1.36 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.39 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 8, lower bound: -0.0115195, upper bound: 0.0115636
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 8, lower bound: -0.0115564, upper bound: 0.0115331
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 8, lower bound: -0.0112625, upper bound: 0.0113264
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 8, lower bound: -0.0113178, upper bound: 0.0112659
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 8, lower bound: -0.0121347, upper bound: 0.0121383
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 8, lower bound: -0.0120790, upper bound: 0.0121778
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 8, lower bound: -0.0120910, upper bound: 0.0120999
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 8, lower bound: -0.0120910, upper bound: 0.0121978
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 8, lower bound: -0.0115632, upper bound: 0.0116713
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 8, lower bound: -0.0115632, upper bound: 0.0116713
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 8, lower bound: -0.0118989, upper bound: 0.0116182
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 8, lower bound: -0.0119300, upper bound: 0.0115815
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 8, lower bound: -0.0114565, upper bound: 0.0115863
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 8, lower bound: -0.0115915, upper bound: 0.0114481
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 8, lower bound: -0.0115376, upper bound: 0.0115638
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 8, lower bound: -0.0115748, upper bound: 0.0115283

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0188877, 0.0188877

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113350, upper bound: 0.0113223
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112810, upper bound: 0.0113845
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0188877, 0.0188877

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111439, upper bound: 0.0112263
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112561, upper bound: 0.0111328
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0187259, 0.0188721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111117, upper bound: 0.0111686
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111117, upper bound: 0.0111686
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0187473, 0.0188497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111480, upper bound: 0.0111346
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111769, upper bound: 0.0111010
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0188877, 0.0188877

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116617, upper bound: 0.0118985
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118855, upper bound: 0.0116716
time: 1.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0188877, 0.0188877

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117872, upper bound: 0.0119241
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118125, upper bound: 0.0118809
time: 1.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0184987, 0.0185669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114036, upper bound: 0.0113410
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114036, upper bound: 0.0113410
time: 1.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0185654, 0.0185045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118758, upper bound: 0.0119483
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118098, upper bound: 0.0119861
time: 1.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0178667, 0.0178122

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109573, upper bound: 0.0110162
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109573, upper bound: 0.0110162
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0178238, 0.0179258

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111617, upper bound: 0.0113899
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112603, upper bound: 0.0112812
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0174949, 0.0175501

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117988, upper bound: 0.0115287
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118053, upper bound: 0.0115146
time: 1.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0175222, 0.0175298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117576, upper bound: 0.0114185
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117576, upper bound: 0.0114185
time: 1.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0186073, 0.0185237

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113044, upper bound: 0.0114446
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113217, upper bound: 0.0114218
time: 1.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0185312, 0.0186031

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108467, upper bound: 0.0107374
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108467, upper bound: 0.0107374
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0188877, 0.0188877

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110954, upper bound: 0.0111557
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111399, upper bound: 0.0111062
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0188877, 0.0188877

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110939, upper bound: 0.0112113
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112742, upper bound: 0.0110678
time: 1.34 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.38 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 8, lower bound: -0.0113350, upper bound: 0.0113223
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 8, lower bound: -0.0112810, upper bound: 0.0113845
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 8, lower bound: -0.0111439, upper bound: 0.0112263
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 8, lower bound: -0.0112561, upper bound: 0.0111328
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 8, lower bound: -0.0111117, upper bound: 0.0111686
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 8, lower bound: -0.0111117, upper bound: 0.0111686
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 8, lower bound: -0.0111480, upper bound: 0.0111346
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 8, lower bound: -0.0111769, upper bound: 0.0111010
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 8, lower bound: -0.0116617, upper bound: 0.0118985
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 8, lower bound: -0.0118855, upper bound: 0.0116716
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 8, lower bound: -0.0117872, upper bound: 0.0119241
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 8, lower bound: -0.0118125, upper bound: 0.0118809
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 8, lower bound: -0.0114036, upper bound: 0.0113410
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 8, lower bound: -0.0114036, upper bound: 0.0113410
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 8, lower bound: -0.0118758, upper bound: 0.0119483
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 8, lower bound: -0.0118098, upper bound: 0.0119861
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 8, lower bound: -0.0109573, upper bound: 0.0110162
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 8, lower bound: -0.0109573, upper bound: 0.0110162
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 8, lower bound: -0.0111617, upper bound: 0.0113899
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 8, lower bound: -0.0112603, upper bound: 0.0112812
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 8, lower bound: -0.0117988, upper bound: 0.0115287
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 8, lower bound: -0.0118053, upper bound: 0.0115146
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 8, lower bound: -0.0117576, upper bound: 0.0114185
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 8, lower bound: -0.0117576, upper bound: 0.0114185
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 8, lower bound: -0.0113044, upper bound: 0.0114446
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 8, lower bound: -0.0113217, upper bound: 0.0114218
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 8, lower bound: -0.0108467, upper bound: 0.0107374
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 8, lower bound: -0.0108467, upper bound: 0.0107374
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 8, lower bound: -0.0110954, upper bound: 0.0111557
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 8, lower bound: -0.0111399, upper bound: 0.0111062
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 8, lower bound: -0.0110939, upper bound: 0.0112113
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 8, lower bound: -0.0112742, upper bound: 0.0110678

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0185464, 0.0185916

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109331, upper bound: 0.0110194
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110251, upper bound: 0.0109166
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0186051, 0.0185246

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108324, upper bound: 0.0110972
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109901, upper bound: 0.0109332
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0176633, 0.0176279

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110039, upper bound: 0.0110695
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110039, upper bound: 0.0110690
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0176212, 0.0176597

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108389, upper bound: 0.0107753
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108878, upper bound: 0.0107231
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0185114, 0.0187685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106764, upper bound: 0.0107206
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106771, upper bound: 0.0107174
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0186095, 0.0186577

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109340, upper bound: 0.0109278
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108727, upper bound: 0.0109910
time: 1.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0185546, 0.0186380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107008, upper bound: 0.0106982
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107039, upper bound: 0.0106960
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0185346, 0.0186584

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107948, upper bound: 0.0108141
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108926, upper bound: 0.0107359
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0174973, 0.0175484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108181, upper bound: 0.0109172
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108181, upper bound: 0.0109172
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0174589, 0.0175677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118519, upper bound: 0.0116107
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118286, upper bound: 0.0116387
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0186326, 0.0187157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108768, upper bound: 0.0109045
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108768, upper bound: 0.0109045
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0186389, 0.0186883

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113461, upper bound: 0.0116232
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115729, upper bound: 0.0114144
time: 1.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0184111, 0.0184569

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112568, upper bound: 0.0111852
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112568, upper bound: 0.0111852
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0183891, 0.0185669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109872, upper bound: 0.0109240
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109888, upper bound: 0.0109226
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0185344, 0.0184812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114083, upper bound: 0.0117069
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116281, upper bound: 0.0114869
time: 1.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0185422, 0.0184743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114414, upper bound: 0.0117157
time: 2.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115536, upper bound: 0.0115887
time: 1.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0178193, 0.0177704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108253, upper bound: 0.0108745
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108253, upper bound: 0.0108750
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0178250, 0.0178122

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105957, upper bound: 0.0107271
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106516, upper bound: 0.0106414
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0172751, 0.0173988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110953, upper bound: 0.0113200
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110792, upper bound: 0.0113246
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0172958, 0.0173778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111552, upper bound: 0.0111817
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111572, upper bound: 0.0111741
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0171122, 0.0171856

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114392, upper bound: 0.0112469
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114392, upper bound: 0.0112469
time: 1.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0171303, 0.0171628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109046, upper bound: 0.0107244
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109046, upper bound: 0.0107244
time: 1.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0174665, 0.0174879

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106726, upper bound: 0.0105102
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106726, upper bound: 0.0105107
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0174803, 0.0175298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105933, upper bound: 0.0105114
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105933, upper bound: 0.0105114
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0184271, 0.0183247

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109819, upper bound: 0.0111630
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110270, upper bound: 0.0111218
time: 1.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0184083, 0.0183404

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108638, upper bound: 0.0110166
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109023, upper bound: 0.0109732
time: 1.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0183167, 0.0184876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108037, upper bound: 0.0106894
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108014, upper bound: 0.0106961
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0184276, 0.0183887

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104597, upper bound: 0.0103833
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104597, upper bound: 0.0103833
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0184591, 0.0184788

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106171, upper bound: 0.0108540
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107853, upper bound: 0.0106609
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0184696, 0.0184566

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109510, upper bound: 0.0108661
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108931, upper bound: 0.0109180
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0176261, 0.0176142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110251, upper bound: 0.0111380
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110105, upper bound: 0.0111411
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0176610, 0.0175878

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111435, upper bound: 0.0109554
time: 2.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111858, upper bound: 0.0109552
time: 1.21 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 5.22 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0109331, upper bound: 0.0110194
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0110251, upper bound: 0.0109166
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0108324, upper bound: 0.0110972
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0109901, upper bound: 0.0109332
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0110039, upper bound: 0.0110695
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0110039, upper bound: 0.0110690
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0108389, upper bound: 0.0107753
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0108878, upper bound: 0.0107231
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0106764, upper bound: 0.0107206
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0106771, upper bound: 0.0107174
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0109340, upper bound: 0.0109278
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0108727, upper bound: 0.0109910
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0107008, upper bound: 0.0106982
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0107039, upper bound: 0.0106960
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0107948, upper bound: 0.0108141
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0108926, upper bound: 0.0107359
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0108181, upper bound: 0.0109172
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0108181, upper bound: 0.0109172
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0118519, upper bound: 0.0116107
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0118286, upper bound: 0.0116387
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0108768, upper bound: 0.0109045
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0108768, upper bound: 0.0109045
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0113461, upper bound: 0.0116232
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0115729, upper bound: 0.0114144
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0112568, upper bound: 0.0111852
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0112568, upper bound: 0.0111852
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0109872, upper bound: 0.0109240
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0109888, upper bound: 0.0109226
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0114083, upper bound: 0.0117069
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0116281, upper bound: 0.0114869
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0114414, upper bound: 0.0117157
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0115536, upper bound: 0.0115887
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0108253, upper bound: 0.0108745
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0108253, upper bound: 0.0108750
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0105957, upper bound: 0.0107271
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0106516, upper bound: 0.0106414
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0110953, upper bound: 0.0113200
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0110792, upper bound: 0.0113246
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0111552, upper bound: 0.0111817
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0111572, upper bound: 0.0111741
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0114392, upper bound: 0.0112469
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0114392, upper bound: 0.0112469
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0109046, upper bound: 0.0107244
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0109046, upper bound: 0.0107244
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0106726, upper bound: 0.0105102
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0106726, upper bound: 0.0105107
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0105933, upper bound: 0.0105114
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0105933, upper bound: 0.0105114
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0109819, upper bound: 0.0111630
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0110270, upper bound: 0.0111218
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0108638, upper bound: 0.0110166
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0109023, upper bound: 0.0109732
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0108037, upper bound: 0.0106894
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0108014, upper bound: 0.0106961
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0104597, upper bound: 0.0103833
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0104597, upper bound: 0.0103833
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0106171, upper bound: 0.0108540
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0107853, upper bound: 0.0106609
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0109510, upper bound: 0.0108661
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0108931, upper bound: 0.0109180
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0110251, upper bound: 0.0111380
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0110105, upper bound: 0.0111411
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0111435, upper bound: 0.0109554
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 8, lower bound: -0.0111858, upper bound: 0.0109552

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0171081, 0.0170848

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105044, upper bound: 0.0107127
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106338, upper bound: 0.0105817
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0170396, 0.0171076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108774, upper bound: 0.0107726
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108774, upper bound: 0.0107726
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0172519, 0.0171632

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107231, upper bound: 0.0110024
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107296, upper bound: 0.0109792
time: 1.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0172437, 0.0171283

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106089, upper bound: 0.0106382
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107055, upper bound: 0.0105666
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0174982, 0.0175704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105920, upper bound: 0.0107091
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106210, upper bound: 0.0106551
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0175746, 0.0174628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109632, upper bound: 0.0110173
time: 2.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109433, upper bound: 0.0110319
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0170515, 0.0171287

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108018, upper bound: 0.0107217
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107832, upper bound: 0.0107384
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0170902, 0.0171169

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104421, upper bound: 0.0103042
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104427, upper bound: 0.0103007
time: 1.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0180979, 0.0183743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102903, upper bound: 0.0104371
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0103833, upper bound: 0.0103194
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0181162, 0.0183590

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104969, upper bound: 0.0104704
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104304, upper bound: 0.0105418
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0180646, 0.0181810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105985, upper bound: 0.0105892
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105985, upper bound: 0.0105912
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0181300, 0.0181140

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105070, upper bound: 0.0106977
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105816, upper bound: 0.0106255
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0181265, 0.0182435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0103264, upper bound: 0.0104133
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104156, upper bound: 0.0103251
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0181603, 0.0182269

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105268, upper bound: 0.0105150
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105268, upper bound: 0.0105150
time: 1.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0177455, 0.0177858

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106220, upper bound: 0.0105861
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105457, upper bound: 0.0106223
time: 1.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0176667, 0.0178609

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104380, upper bound: 0.0103004
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104383, upper bound: 0.0102968
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0174263, 0.0174361

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104237, upper bound: 0.0105620
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104605, upper bound: 0.0105226
time: 1.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0173845, 0.0175484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106927, upper bound: 0.0107850
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106927, upper bound: 0.0107845
time: 1.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0173020, 0.0174602

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116669, upper bound: 0.0114686
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117325, upper bound: 0.0114582
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0173613, 0.0174108

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116924, upper bound: 0.0115115
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116924, upper bound: 0.0115119
time: 1.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0186225, 0.0187054

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107246, upper bound: 0.0107724
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107391, upper bound: 0.0107672
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0186224, 0.0187157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107919, upper bound: 0.0108252
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107896, upper bound: 0.0108268
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0171222, 0.0171303

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109442, upper bound: 0.0113584
time: 4.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111127, upper bound: 0.0112372
time: 1.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0170841, 0.0171596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111908, upper bound: 0.0111325
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113067, upper bound: 0.0110329
time: 1.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0182025, 0.0183369

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108130, upper bound: 0.0107487
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108132, upper bound: 0.0107469
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0182758, 0.0182484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107968, upper bound: 0.0108719
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109615, upper bound: 0.0107600
time: 1.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0179842, 0.0181879

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109504, upper bound: 0.0108639
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109283, upper bound: 0.0108893
time: 1.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0180130, 0.0181782

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105840, upper bound: 0.0106216
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106844, upper bound: 0.0105301
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0169941, 0.0169150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112669, upper bound: 0.0115710
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112667, upper bound: 0.0115710
time: 1.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0169709, 0.0169529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111766, upper bound: 0.0112087
time: 1.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113563, upper bound: 0.0110311
time: 1.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0180041, 0.0179447

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105811, upper bound: 0.0106907
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105811, upper bound: 0.0106907
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0180133, 0.0179119

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110868, upper bound: 0.0113286
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113147, upper bound: 0.0111274
time: 1.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0176542, 0.0177161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0103963, upper bound: 0.0105519
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105005, upper bound: 0.0104414
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0177295, 0.0176053

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104070, upper bound: 0.0104978
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104473, upper bound: 0.0104530
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0171383, 0.0169778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101620, upper bound: 0.0102909
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101620, upper bound: 0.0102909
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0169915, 0.0170533

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105196, upper bound: 0.0105024
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105196, upper bound: 0.0105024
time: 1.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0171099, 0.0173445

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106909, upper bound: 0.0110549
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107661, upper bound: 0.0108338
time: 1.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0172071, 0.0172337

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110396, upper bound: 0.0112759
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110024, upper bound: 0.0112871
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0168855, 0.0169870

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104816, upper bound: 0.0105173
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104816, upper bound: 0.0105173
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0169063, 0.0169377

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108087, upper bound: 0.0108713
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108603, upper bound: 0.0108501
time: 1.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0171026, 0.0171761

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110061, upper bound: 0.0109163
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110999, upper bound: 0.0108376
time: 1.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0171027, 0.0171856

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112455, upper bound: 0.0110979
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113054, upper bound: 0.0110927
time: 1.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0170651, 0.0171015

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108203, upper bound: 0.0106356
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108200, upper bound: 0.0106372
time: 1.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0170690, 0.0171628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105215, upper bound: 0.0104472
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106166, upper bound: 0.0103455
time: 1.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0174582, 0.0174787

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105924, upper bound: 0.0104219
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105912, upper bound: 0.0104233
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0174573, 0.0174879

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105507, upper bound: 0.0103986
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105685, upper bound: 0.0103951
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0173896, 0.0174157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104882, upper bound: 0.0104070
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104914, upper bound: 0.0104065
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0173659, 0.0175298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104288, upper bound: 0.0103674
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104288, upper bound: 0.0103674
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0180152, 0.0179354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108763, upper bound: 0.0110714
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108774, upper bound: 0.0110389
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0180379, 0.0179098

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100649, upper bound: 0.0101313
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100649, upper bound: 0.0101313
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0178413, 0.0178026

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107494, upper bound: 0.0109059
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107494, upper bound: 0.0109061
time: 1.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0026925, 0.0222649, 0.0026925, 0.0222649, -0.0195724, 0.0195724
1: -0.0048843, 0.0040999, -0.0048843, 0.0040999, -0.0089842, 0.0089842
2: -0.0005510, 0.0128712, -0.0005510, 0.0128712, -0.0134222, 0.0134222
3: -0.0070301, 0.0042693, -0.0070301, 0.0042693, -0.0112994, 0.0112994
4: -0.0033761, 0.0021978, -0.0033761, 0.0021978, -0.0055739, 0.0055739
5: -0.0026469, 0.0065318, -0.0026469, 0.0065318, -0.0091787, 0.0091787
6: -0.0178152, 0.0036158, -0.0178152, 0.0036158, -0.0214309, 0.0214309
7: -0.0142761, 0.0178190, -0.0142761, 0.0178190, -0.0320951, 0.0320951
8: 0.9815300, 1.0025511, 0.9815300, 1.0025511, -0.0210211, 0.0210211
9: -0.0168554, 0.0020323, -0.0168554, 0.0020323, -0.0178704, 0.0177847

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107880, upper bound: 0.0108708
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107908, upper bound: 0.0108560
time: 1.16 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 6.08 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0105044, upper bound: 0.0107127
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0106338, upper bound: 0.0105817
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0108774, upper bound: 0.0107726
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0108774, upper bound: 0.0107726
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0107231, upper bound: 0.0110024
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0107296, upper bound: 0.0109792
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0106089, upper bound: 0.0106382
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0107055, upper bound: 0.0105666
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0105920, upper bound: 0.0107091
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0106210, upper bound: 0.0106551
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0109632, upper bound: 0.0110173
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0109433, upper bound: 0.0110319
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0108018, upper bound: 0.0107217
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0107832, upper bound: 0.0107384
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0104421, upper bound: 0.0103042
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0104427, upper bound: 0.0103007
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0102903, upper bound: 0.0104371
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0103833, upper bound: 0.0103194
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0104969, upper bound: 0.0104704
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0104304, upper bound: 0.0105418
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0105985, upper bound: 0.0105892
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0105985, upper bound: 0.0105912
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0105070, upper bound: 0.0106977
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0105816, upper bound: 0.0106255
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0103264, upper bound: 0.0104133
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0104156, upper bound: 0.0103251
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0105268, upper bound: 0.0105150
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0105268, upper bound: 0.0105150
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0106220, upper bound: 0.0105861
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0105457, upper bound: 0.0106223
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0104380, upper bound: 0.0103004
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0104383, upper bound: 0.0102968
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0104237, upper bound: 0.0105620
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0104605, upper bound: 0.0105226
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0106927, upper bound: 0.0107850
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0106927, upper bound: 0.0107845
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0116669, upper bound: 0.0114686
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0117325, upper bound: 0.0114582
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0116924, upper bound: 0.0115115
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0116924, upper bound: 0.0115119
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0107246, upper bound: 0.0107724
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0107391, upper bound: 0.0107672
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0107919, upper bound: 0.0108252
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0107896, upper bound: 0.0108268
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0109442, upper bound: 0.0113584
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0111127, upper bound: 0.0112372
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0111908, upper bound: 0.0111325
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0113067, upper bound: 0.0110329
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0108130, upper bound: 0.0107487
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0108132, upper bound: 0.0107469
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0107968, upper bound: 0.0108719
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0109615, upper bound: 0.0107600
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0109504, upper bound: 0.0108639
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0109283, upper bound: 0.0108893
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0105840, upper bound: 0.0106216
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0106844, upper bound: 0.0105301
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0112669, upper bound: 0.0115710
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0112667, upper bound: 0.0115710
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0111766, upper bound: 0.0112087
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0113563, upper bound: 0.0110311
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0105811, upper bound: 0.0106907
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0105811, upper bound: 0.0106907
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0110868, upper bound: 0.0113286
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0113147, upper bound: 0.0111274
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0103963, upper bound: 0.0105519
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0105005, upper bound: 0.0104414
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0104070, upper bound: 0.0104978
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0104473, upper bound: 0.0104530
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0101620, upper bound: 0.0102909
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0101620, upper bound: 0.0102909
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0105196, upper bound: 0.0105024
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0105196, upper bound: 0.0105024
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0106909, upper bound: 0.0110549
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0107661, upper bound: 0.0108338
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0110396, upper bound: 0.0112759
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0110024, upper bound: 0.0112871
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0104816, upper bound: 0.0105173
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0104816, upper bound: 0.0105173
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0108087, upper bound: 0.0108713
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0108603, upper bound: 0.0108501
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0110061, upper bound: 0.0109163
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0110999, upper bound: 0.0108376
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0112455, upper bound: 0.0110979
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0113054, upper bound: 0.0110927
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0108203, upper bound: 0.0106356
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0108200, upper bound: 0.0106372
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0105215, upper bound: 0.0104472
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0106166, upper bound: 0.0103455
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0105924, upper bound: 0.0104219
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0105912, upper bound: 0.0104233
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0105507, upper bound: 0.0103986
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0105685, upper bound: 0.0103951
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0104882, upper bound: 0.0104070
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0104914, upper bound: 0.0104065
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0104288, upper bound: 0.0103674
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0104288, upper bound: 0.0103674
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0108763, upper bound: 0.0110714
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0108774, upper bound: 0.0110389
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0100649, upper bound: 0.0101313
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0100649, upper bound: 0.0101313
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0107494, upper bound: 0.0109059
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0107494, upper bound: 0.0109061
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0107880, upper bound: 0.0108708
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 8, lower bound: -0.0107908, upper bound: 0.0108560
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.08
Output dim: 8, lower bound: -0.0108037, upper bound: 0.0106894
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.08
Output dim: 8, lower bound: -0.0108014, upper bound: 0.0106961
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.08
Output dim: 8, lower bound: -0.0104597, upper bound: 0.0103833
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.08
Output dim: 8, lower bound: -0.0104597, upper bound: 0.0103833
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.08
Output dim: 8, lower bound: -0.0106171, upper bound: 0.0108540
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.08
Output dim: 8, lower bound: -0.0107853, upper bound: 0.0106609
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.08
Output dim: 8, lower bound: -0.0109510, upper bound: 0.0108661
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.08
Output dim: 8, lower bound: -0.0108931, upper bound: 0.0109180
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.08
Output dim: 8, lower bound: -0.0110251, upper bound: 0.0111380
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.08
Output dim: 8, lower bound: -0.0110105, upper bound: 0.0111411
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.08
Output dim: 8, lower bound: -0.0111435, upper bound: 0.0109554
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.08
Output dim: 8, lower bound: -0.0111858, upper bound: 0.0109552

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 4.06 + 597.56 = 601.62 seconds
