## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.04893569308


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171)
1: (-0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138)
2: (-0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719)
3: (-0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281)
4: (-0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168)
5: (-0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703)
6: (-0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523)
7: (-0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926)
8: (0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126)
9: (-0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.39 + 2.31 = 3.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.1481142, upper bound: 0.1481142

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1417946, upper bound: 0.1417946
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1417946, upper bound: 0.1417946
time: 1.16 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.31 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.31
Output dim: 8, lower bound: -0.1417946, upper bound: 0.1417946
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.31
Output dim: 8, lower bound: -0.1417946, upper bound: 0.1417946

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1377326, upper bound: 0.1377326
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1377326, upper bound: 0.1377326
time: 1.26 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1412508, upper bound: 0.1413138
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1413138, upper bound: 0.1412508
time: 1.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.71 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.71
Output dim: 8, lower bound: -0.1377326, upper bound: 0.1377326
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.71
Output dim: 8, lower bound: -0.1377326, upper bound: 0.1377326
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.71
Output dim: 8, lower bound: -0.1412508, upper bound: 0.1413138
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.71
Output dim: 8, lower bound: -0.1413138, upper bound: 0.1412508

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1375321, upper bound: 0.1375969
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1375969, upper bound: 0.1375321
time: 1.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1372853, upper bound: 0.1374752
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1374752, upper bound: 0.1372853
time: 1.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1378964, upper bound: 0.1379463
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1378964, upper bound: 0.1379463
time: 1.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1411810, upper bound: 0.1408881
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1409594, upper bound: 0.1411035
time: 1.36 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.29 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.29
Output dim: 8, lower bound: -0.1375321, upper bound: 0.1375969
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.29
Output dim: 8, lower bound: -0.1375969, upper bound: 0.1375321
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.29
Output dim: 8, lower bound: -0.1372853, upper bound: 0.1374752
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.29
Output dim: 8, lower bound: -0.1374752, upper bound: 0.1372853
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.29
Output dim: 8, lower bound: -0.1378964, upper bound: 0.1379463
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.29
Output dim: 8, lower bound: -0.1378964, upper bound: 0.1379463
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.29
Output dim: 8, lower bound: -0.1411810, upper bound: 0.1408881
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.29
Output dim: 8, lower bound: -0.1409594, upper bound: 0.1411035

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1335904, upper bound: 0.1336640
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1335904, upper bound: 0.1336640
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1238680, upper bound: 0.1238176
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1238680, upper bound: 0.1238176
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1363393, upper bound: 0.1366161
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1364233, upper bound: 0.1365165
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1328548, upper bound: 0.1327111
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1328543, upper bound: 0.1327112
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1373619, upper bound: 0.1375932
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1375529, upper bound: 0.1374405
time: 1.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1378549, upper bound: 0.1379028
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1378532, upper bound: 0.1379049
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1375306, upper bound: 0.1372676
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1375306, upper bound: 0.1372676
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1270109, upper bound: 0.1271097
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1270109, upper bound: 0.1271097
time: 0.88 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.00 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 8, lower bound: -0.1335904, upper bound: 0.1336640
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 8, lower bound: -0.1335904, upper bound: 0.1336640
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 8, lower bound: -0.1238680, upper bound: 0.1238176
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 8, lower bound: -0.1238680, upper bound: 0.1238176
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 8, lower bound: -0.1363393, upper bound: 0.1366161
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 8, lower bound: -0.1364233, upper bound: 0.1365165
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 8, lower bound: -0.1328548, upper bound: 0.1327111
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 8, lower bound: -0.1328543, upper bound: 0.1327112
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 8, lower bound: -0.1373619, upper bound: 0.1375932
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 8, lower bound: -0.1375529, upper bound: 0.1374405
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 8, lower bound: -0.1378549, upper bound: 0.1379028
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 8, lower bound: -0.1378532, upper bound: 0.1379049
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 8, lower bound: -0.1375306, upper bound: 0.1372676
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 8, lower bound: -0.1375306, upper bound: 0.1372676
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 8, lower bound: -0.1270109, upper bound: 0.1271097
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 8, lower bound: -0.1270109, upper bound: 0.1271097

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1326012, upper bound: 0.1327717
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1326972, upper bound: 0.1326463
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1335903, upper bound: 0.1336635
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1335904, upper bound: 0.1336640
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 169

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1237726, upper bound: 0.1236873
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1237278, upper bound: 0.1237209
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1156794, upper bound: 0.1156949
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1157288, upper bound: 0.1156341
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1314754, upper bound: 0.1325154
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1322363, upper bound: 0.1317575
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1360484, upper bound: 0.1361199
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1360477, upper bound: 0.1361199
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1326602, upper bound: 0.1325594
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1327027, upper bound: 0.1325033
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1328542, upper bound: 0.1327112
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1328543, upper bound: 0.1327111
time: 1.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1373204, upper bound: 0.1375498
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1373143, upper bound: 0.1375518
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1132959, upper bound: 0.1132803
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1132959, upper bound: 0.1132803
time: 1.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1324405, upper bound: 0.1331971
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1331462, upper bound: 0.1324157
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1364397, upper bound: 0.1368833
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1368323, upper bound: 0.1365561
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1159107, upper bound: 0.1158440
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1159107, upper bound: 0.1158440
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1327686, upper bound: 0.1335433
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1338038, upper bound: 0.1325042
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1260383, upper bound: 0.1261995
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1260958, upper bound: 0.1261006
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1265807, upper bound: 0.1266993
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1266050, upper bound: 0.1266738
time: 0.74 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.79 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 8, lower bound: -0.1326012, upper bound: 0.1327717
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 8, lower bound: -0.1326972, upper bound: 0.1326463
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 8, lower bound: -0.1335903, upper bound: 0.1336635
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 8, lower bound: -0.1335904, upper bound: 0.1336640
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 8, lower bound: -0.1237726, upper bound: 0.1236873
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 8, lower bound: -0.1237278, upper bound: 0.1237209
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 8, lower bound: -0.1156794, upper bound: 0.1156949
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 8, lower bound: -0.1157288, upper bound: 0.1156341
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 8, lower bound: -0.1314754, upper bound: 0.1325154
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 8, lower bound: -0.1322363, upper bound: 0.1317575
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 8, lower bound: -0.1360484, upper bound: 0.1361199
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 8, lower bound: -0.1360477, upper bound: 0.1361199
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 8, lower bound: -0.1326602, upper bound: 0.1325594
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 8, lower bound: -0.1327027, upper bound: 0.1325033
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 8, lower bound: -0.1328542, upper bound: 0.1327112
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 8, lower bound: -0.1328543, upper bound: 0.1327111
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 8, lower bound: -0.1373204, upper bound: 0.1375498
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 8, lower bound: -0.1373143, upper bound: 0.1375518
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 8, lower bound: -0.1132959, upper bound: 0.1132803
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 8, lower bound: -0.1132959, upper bound: 0.1132803
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 8, lower bound: -0.1324405, upper bound: 0.1331971
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 8, lower bound: -0.1331462, upper bound: 0.1324157
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 8, lower bound: -0.1364397, upper bound: 0.1368833
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 8, lower bound: -0.1368323, upper bound: 0.1365561
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 8, lower bound: -0.1159107, upper bound: 0.1158440
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 8, lower bound: -0.1159107, upper bound: 0.1158440
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 8, lower bound: -0.1327686, upper bound: 0.1335433
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 8, lower bound: -0.1338038, upper bound: 0.1325042
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 8, lower bound: -0.1260383, upper bound: 0.1261995
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 8, lower bound: -0.1260958, upper bound: 0.1261006
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 8, lower bound: -0.1265807, upper bound: 0.1266993
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 8, lower bound: -0.1266050, upper bound: 0.1266738

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1293748, upper bound: 0.1295397
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1293748, upper bound: 0.1295397
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1326573, upper bound: 0.1326066
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1326572, upper bound: 0.1326066
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1297552, upper bound: 0.1298039
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1297523, upper bound: 0.1298054
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1334674, upper bound: 0.1334178
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1333706, upper bound: 0.1335688
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1207778, upper bound: 0.1207182
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1207778, upper bound: 0.1207182
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1236775, upper bound: 0.1236706
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1236774, upper bound: 0.1236703
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1155813, upper bound: 0.1155290
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1155116, upper bound: 0.1155958
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1144947, upper bound: 0.1144114
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1144942, upper bound: 0.1144114
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276503, upper bound: 0.1285299
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276503, upper bound: 0.1285299
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1315849, upper bound: 0.1310758
time: 1.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1315845, upper bound: 0.1310758
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1315223, upper bound: 0.1315828
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1315223, upper bound: 0.1315828
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1360477, upper bound: 0.1361138
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1360471, upper bound: 0.1361199
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1169175, upper bound: 0.1168726
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1169175, upper bound: 0.1168726
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1326607, upper bound: 0.1324530
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1326606, upper bound: 0.1324615
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1326565, upper bound: 0.1324997
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1326564, upper bound: 0.1325090
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1326594, upper bound: 0.1325594
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1327023, upper bound: 0.1325036
time: 1.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1219808, upper bound: 0.1220312
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1219808, upper bound: 0.1220312
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1319796, upper bound: 0.1328499
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1325809, upper bound: 0.1320806
time: 1.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1132156, upper bound: 0.1131837
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1132023, upper bound: 0.1132007
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 169

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1062578, upper bound: 0.1063302
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1063440, upper bound: 0.1062582
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1284749, upper bound: 0.1290416
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1284749, upper bound: 0.1290416
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1321964, upper bound: 0.1315536
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1322964, upper bound: 0.1314414
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1258049, upper bound: 0.1259358
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1258049, upper bound: 0.1259358
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 169

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1368323, upper bound: 0.1365457
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1368286, upper bound: 0.1365561
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1146963, upper bound: 0.1146225
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1146963, upper bound: 0.1146225
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1146852, upper bound: 0.1146448
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1147105, upper bound: 0.1146144
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1285017, upper bound: 0.1290235
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1285017, upper bound: 0.1290235
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1322232, upper bound: 0.1313873
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1326271, upper bound: 0.1307190
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1259908, upper bound: 0.1261403
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1259763, upper bound: 0.1261522
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1249375, upper bound: 0.1249580
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1249375, upper bound: 0.1249580
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1264778, upper bound: 0.1266013
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1264718, upper bound: 0.1266015
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1256279, upper bound: 0.1257628
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1256907, upper bound: 0.1256799
time: 1.03 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.25 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1293748, upper bound: 0.1295397
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1293748, upper bound: 0.1295397
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1326573, upper bound: 0.1326066
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1326572, upper bound: 0.1326066
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1297552, upper bound: 0.1298039
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1297523, upper bound: 0.1298054
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1334674, upper bound: 0.1334178
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1333706, upper bound: 0.1335688
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1207778, upper bound: 0.1207182
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1207778, upper bound: 0.1207182
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1236775, upper bound: 0.1236706
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1236774, upper bound: 0.1236703
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1155813, upper bound: 0.1155290
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1155116, upper bound: 0.1155958
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1144947, upper bound: 0.1144114
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1144942, upper bound: 0.1144114
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1276503, upper bound: 0.1285299
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1276503, upper bound: 0.1285299
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1315849, upper bound: 0.1310758
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1315845, upper bound: 0.1310758
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1315223, upper bound: 0.1315828
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1315223, upper bound: 0.1315828
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1360477, upper bound: 0.1361138
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1360471, upper bound: 0.1361199
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1169175, upper bound: 0.1168726
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1169175, upper bound: 0.1168726
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1326607, upper bound: 0.1324530
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1326606, upper bound: 0.1324615
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1326565, upper bound: 0.1324997
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1326564, upper bound: 0.1325090
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1326594, upper bound: 0.1325594
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1327023, upper bound: 0.1325036
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1219808, upper bound: 0.1220312
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1219808, upper bound: 0.1220312
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1319796, upper bound: 0.1328499
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1325809, upper bound: 0.1320806
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1132156, upper bound: 0.1131837
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1132023, upper bound: 0.1132007
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1062578, upper bound: 0.1063302
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1063440, upper bound: 0.1062582
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1284749, upper bound: 0.1290416
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1284749, upper bound: 0.1290416
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1321964, upper bound: 0.1315536
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1322964, upper bound: 0.1314414
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1258049, upper bound: 0.1259358
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1258049, upper bound: 0.1259358
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1368323, upper bound: 0.1365457
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1368286, upper bound: 0.1365561
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1146963, upper bound: 0.1146225
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1146963, upper bound: 0.1146225
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1146852, upper bound: 0.1146448
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1147105, upper bound: 0.1146144
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1285017, upper bound: 0.1290235
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1285017, upper bound: 0.1290235
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1322232, upper bound: 0.1313873
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1326271, upper bound: 0.1307190
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1259908, upper bound: 0.1261403
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1259763, upper bound: 0.1261522
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1249375, upper bound: 0.1249580
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1249375, upper bound: 0.1249580
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1264778, upper bound: 0.1266013
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1264718, upper bound: 0.1266015
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1256279, upper bound: 0.1257628
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 8, lower bound: -0.1256907, upper bound: 0.1256799

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1269290, upper bound: 0.1270774
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1269284, upper bound: 0.1270797
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1086946, upper bound: 0.1087513
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1086946, upper bound: 0.1087513
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 169

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1326456, upper bound: 0.1325716
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1326359, upper bound: 0.1325954
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1321115, upper bound: 0.1321161
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1321660, upper bound: 0.1320871
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1122248, upper bound: 0.1122615
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1122248, upper bound: 0.1122615
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 169

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1297425, upper bound: 0.1297831
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1297351, upper bound: 0.1297958
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1333564, upper bound: 0.1332620
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1333202, upper bound: 0.1333069
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1323482, upper bound: 0.1325247
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1323482, upper bound: 0.1325259
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1116816, upper bound: 0.1116353
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1116816, upper bound: 0.1116353
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1207023, upper bound: 0.1206388
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1207001, upper bound: 0.1206422
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1235995, upper bound: 0.1235918
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1235996, upper bound: 0.1235950
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1236774, upper bound: 0.1236703
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1236766, upper bound: 0.1236703
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1095351, upper bound: 0.1094679
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1095351, upper bound: 0.1094679
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1154610, upper bound: 0.1155398
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1154606, upper bound: 0.1155450
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1129403, upper bound: 0.1129754
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1130388, upper bound: 0.1129003
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 169

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1075236, upper bound: 0.1074969
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1075236, upper bound: 0.1074969
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276503, upper bound: 0.1285299
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276499, upper bound: 0.1285299
time: 1.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1271631, upper bound: 0.1280449
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1271635, upper bound: 0.1279893
time: 1.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1110936, upper bound: 0.1109737
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1110936, upper bound: 0.1109737
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1303343, upper bound: 0.1299717
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1304741, upper bound: 0.1295223
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1302297, upper bound: 0.1304641
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1304004, upper bound: 0.1302276
time: 1.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1302297, upper bound: 0.1304641
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1304004, upper bound: 0.1302276
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 169

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1360342, upper bound: 0.1360967
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1360246, upper bound: 0.1361012
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1359209, upper bound: 0.1358623
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1357720, upper bound: 0.1359989
time: 1.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1158084, upper bound: 0.1157813
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1158256, upper bound: 0.1157482
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1154155, upper bound: 0.1154095
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1154553, upper bound: 0.1153670
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1324608, upper bound: 0.1322457
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1324608, upper bound: 0.1322486
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1323385, upper bound: 0.1321405
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1323385, upper bound: 0.1321405
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1275336, upper bound: 0.1278702
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1280497, upper bound: 0.1274585
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1326142, upper bound: 0.1324599
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1326139, upper bound: 0.1324668
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1317081, upper bound: 0.1316785
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1317782, upper bound: 0.1316101
time: 1.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1323815, upper bound: 0.1321826
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1323815, upper bound: 0.1321826
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1218785, upper bound: 0.1218587
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1217903, upper bound: 0.1219336
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1219781, upper bound: 0.1220312
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1219808, upper bound: 0.1220310
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1304706, upper bound: 0.1317510
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1308660, upper bound: 0.1314789
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1312677, upper bound: 0.1309739
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1314832, upper bound: 0.1305624
time: 1.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1119631, upper bound: 0.1119325
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1119631, upper bound: 0.1119326
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1131191, upper bound: 0.1130711
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1130694, upper bound: 0.1131177
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1050187, upper bound: 0.1050946
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1050193, upper bound: 0.1050946
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1051065, upper bound: 0.1050197
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1051085, upper bound: 0.1050192
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1280847, upper bound: 0.1285872
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1280847, upper bound: 0.1285872
time: 1.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1113137, upper bound: 0.1113113
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1113137, upper bound: 0.1113113
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1285383, upper bound: 0.1279749
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1285383, upper bound: 0.1279749
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1322097, upper bound: 0.1312817
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1321983, upper bound: 0.1312843
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1225145, upper bound: 0.1226018
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1225137, upper bound: 0.1226073
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1247333, upper bound: 0.1248789
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1247453, upper bound: 0.1248270
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1362700, upper bound: 0.1359792
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1362695, upper bound: 0.1359803
time: 1.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1367402, upper bound: 0.1364007
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1367338, upper bound: 0.1364422
time: 1.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1116085, upper bound: 0.1115248
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1116085, upper bound: 0.1115248
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1146453, upper bound: 0.1145728
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1146467, upper bound: 0.1145517
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1120962, upper bound: 0.1120691
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1120962, upper bound: 0.1120691
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1084361, upper bound: 0.1085011
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1086003, upper bound: 0.1083651
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1273711, upper bound: 0.1281034
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1275864, upper bound: 0.1280433
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 169

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1284843, upper bound: 0.1289979
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1284254, upper bound: 0.1290120
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 169

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1322033, upper bound: 0.1313320
time: 2.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1321395, upper bound: 0.1313608
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1153716, upper bound: 0.1151226
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1153716, upper bound: 0.1151226
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1259907, upper bound: 0.1261399
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1259908, upper bound: 0.1261403
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1248029, upper bound: 0.1249714
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1248029, upper bound: 0.1249714
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1231874, upper bound: 0.1231670
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1231651, upper bound: 0.1231816
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1205685, upper bound: 0.1206321
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1205685, upper bound: 0.1206321
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1255030, upper bound: 0.1256905
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1255628, upper bound: 0.1255772
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1203755, upper bound: 0.1207873
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1206492, upper bound: 0.1204575
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1255802, upper bound: 0.1257015
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1255634, upper bound: 0.1257153
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 169

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1256769, upper bound: 0.1256202
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1256348, upper bound: 0.1256655
time: 0.97 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.25 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1269290, upper bound: 0.1270774
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1269284, upper bound: 0.1270797
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1086946, upper bound: 0.1087513
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1086946, upper bound: 0.1087513
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1326456, upper bound: 0.1325716
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1326359, upper bound: 0.1325954
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1321115, upper bound: 0.1321161
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1321660, upper bound: 0.1320871
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1122248, upper bound: 0.1122615
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1122248, upper bound: 0.1122615
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1297425, upper bound: 0.1297831
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1297351, upper bound: 0.1297958
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1333564, upper bound: 0.1332620
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1333202, upper bound: 0.1333069
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1323482, upper bound: 0.1325247
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1323482, upper bound: 0.1325259
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1116816, upper bound: 0.1116353
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1116816, upper bound: 0.1116353
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1207023, upper bound: 0.1206388
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1207001, upper bound: 0.1206422
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1235995, upper bound: 0.1235918
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1235996, upper bound: 0.1235950
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1236774, upper bound: 0.1236703
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1236766, upper bound: 0.1236703
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1095351, upper bound: 0.1094679
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1095351, upper bound: 0.1094679
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1154610, upper bound: 0.1155398
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1154606, upper bound: 0.1155450
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1129403, upper bound: 0.1129754
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1130388, upper bound: 0.1129003
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1075236, upper bound: 0.1074969
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1075236, upper bound: 0.1074969
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1276503, upper bound: 0.1285299
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1276499, upper bound: 0.1285299
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1271631, upper bound: 0.1280449
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1271635, upper bound: 0.1279893
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1110936, upper bound: 0.1109737
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1110936, upper bound: 0.1109737
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1303343, upper bound: 0.1299717
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1304741, upper bound: 0.1295223
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1302297, upper bound: 0.1304641
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1304004, upper bound: 0.1302276
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1302297, upper bound: 0.1304641
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1304004, upper bound: 0.1302276
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1360342, upper bound: 0.1360967
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1360246, upper bound: 0.1361012
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1359209, upper bound: 0.1358623
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1357720, upper bound: 0.1359989
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1158084, upper bound: 0.1157813
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1158256, upper bound: 0.1157482
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1154155, upper bound: 0.1154095
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1154553, upper bound: 0.1153670
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1324608, upper bound: 0.1322457
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1324608, upper bound: 0.1322486
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1323385, upper bound: 0.1321405
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1323385, upper bound: 0.1321405
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1275336, upper bound: 0.1278702
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1280497, upper bound: 0.1274585
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1326142, upper bound: 0.1324599
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1326139, upper bound: 0.1324668
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1317081, upper bound: 0.1316785
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1317782, upper bound: 0.1316101
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1323815, upper bound: 0.1321826
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1323815, upper bound: 0.1321826
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1218785, upper bound: 0.1218587
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1217903, upper bound: 0.1219336
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1219781, upper bound: 0.1220312
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1219808, upper bound: 0.1220310
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1304706, upper bound: 0.1317510
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1308660, upper bound: 0.1314789
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1312677, upper bound: 0.1309739
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1314832, upper bound: 0.1305624
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1119631, upper bound: 0.1119325
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1119631, upper bound: 0.1119326
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1131191, upper bound: 0.1130711
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1130694, upper bound: 0.1131177
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1050187, upper bound: 0.1050946
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1050193, upper bound: 0.1050946
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1051065, upper bound: 0.1050197
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1051085, upper bound: 0.1050192
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1280847, upper bound: 0.1285872
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1280847, upper bound: 0.1285872
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1113137, upper bound: 0.1113113
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1113137, upper bound: 0.1113113
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1285383, upper bound: 0.1279749
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1285383, upper bound: 0.1279749
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1322097, upper bound: 0.1312817
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1321983, upper bound: 0.1312843
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1225145, upper bound: 0.1226018
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1225137, upper bound: 0.1226073
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1247333, upper bound: 0.1248789
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1247453, upper bound: 0.1248270
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1362700, upper bound: 0.1359792
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1362695, upper bound: 0.1359803
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1367402, upper bound: 0.1364007
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1367338, upper bound: 0.1364422
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1116085, upper bound: 0.1115248
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1116085, upper bound: 0.1115248
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1146453, upper bound: 0.1145728
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1146467, upper bound: 0.1145517
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1120962, upper bound: 0.1120691
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1120962, upper bound: 0.1120691
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1084361, upper bound: 0.1085011
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1086003, upper bound: 0.1083651
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1273711, upper bound: 0.1281034
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1275864, upper bound: 0.1280433
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1284843, upper bound: 0.1289979
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1284254, upper bound: 0.1290120
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1322033, upper bound: 0.1313320
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1321395, upper bound: 0.1313608
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1153716, upper bound: 0.1151226
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1153716, upper bound: 0.1151226
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1259907, upper bound: 0.1261399
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1259908, upper bound: 0.1261403
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1248029, upper bound: 0.1249714
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1248029, upper bound: 0.1249714
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1231874, upper bound: 0.1231670
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1231651, upper bound: 0.1231816
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1205685, upper bound: 0.1206321
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1205685, upper bound: 0.1206321
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1255030, upper bound: 0.1256905
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1255628, upper bound: 0.1255772
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1203755, upper bound: 0.1207873
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1206492, upper bound: 0.1204575
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1255802, upper bound: 0.1257015
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1255634, upper bound: 0.1257153
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1256769, upper bound: 0.1256202
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.1256348, upper bound: 0.1256655

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1268619, upper bound: 0.1270072
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1268619, upper bound: 0.1270096
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1268617, upper bound: 0.1270090
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1268617, upper bound: 0.1270121
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1086174, upper bound: 0.1086730
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1086171, upper bound: 0.1086701
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1080829, upper bound: 0.1081436
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1080870, upper bound: 0.1081381
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1124822, upper bound: 0.1124618
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1124822, upper bound: 0.1124618
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1325091, upper bound: 0.1323715
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1324187, upper bound: 0.1324683
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1320025, upper bound: 0.1319634
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1319659, upper bound: 0.1320046
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1316846, upper bound: 0.1316065
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1316846, upper bound: 0.1316065
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1121346, upper bound: 0.1121573
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1121224, upper bound: 0.1121752
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1118199, upper bound: 0.1118460
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1118145, upper bound: 0.1118493
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1291716, upper bound: 0.1292645
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1292221, upper bound: 0.1292050
time: 1.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1278908, upper bound: 0.1279611
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1278908, upper bound: 0.1279611
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1328077, upper bound: 0.1327704
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1328689, upper bound: 0.1327042
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 169

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1333056, upper bound: 0.1332912
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1333025, upper bound: 0.1332934
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1317782, upper bound: 0.1320006
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1318260, upper bound: 0.1319559
time: 1.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1312993, upper bound: 0.1315617
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1313880, upper bound: 0.1314556
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1116816, upper bound: 0.1116351
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1116800, upper bound: 0.1116353
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1111037, upper bound: 0.1110563
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1111027, upper bound: 0.1110580
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 169

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1206921, upper bound: 0.1206185
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1206716, upper bound: 0.1206285
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 169

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1206899, upper bound: 0.1206187
time: 1.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1206690, upper bound: 0.1206318
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1171142, upper bound: 0.1171096
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1171142, upper bound: 0.1171096
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1171143, upper bound: 0.1171083
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1171143, upper bound: 0.1171083
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1154851, upper bound: 0.1155452
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1155368, upper bound: 0.1154848
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1206970, upper bound: 0.1206920
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1206970, upper bound: 0.1206920
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 169

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1091038, upper bound: 0.1090513
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1091176, upper bound: 0.1090373
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1094479, upper bound: 0.1093877
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1094548, upper bound: 0.1093845
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1115689, upper bound: 0.1115965
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1115689, upper bound: 0.1115965
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1115689, upper bound: 0.1115973
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1115689, upper bound: 0.1115973
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1055800, upper bound: 0.1055757
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1055800, upper bound: 0.1055757
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0401597, 0.0270574, -0.0401597, 0.0270574, -0.0672171, 0.0672171
1: -0.0272156, 0.0273982, -0.0272156, 0.0273982, -0.0546138, 0.0546138
2: -0.0226134, 0.0609585, -0.0226134, 0.0609585, -0.0835719, 0.0835719
3: -0.0176759, 0.0380522, -0.0176759, 0.0380522, -0.0557281, 0.0557281
4: -0.0434810, 0.0362358, -0.0434810, 0.0362358, -0.0797168, 0.0797168
5: -0.0271673, 0.0757030, -0.0271673, 0.0757030, -0.1028703, 0.1028703
6: -0.0255901, 0.0474622, -0.0255901, 0.0474622, -0.0730523, 0.0730523
7: -0.0574554, 0.0269372, -0.0574554, 0.0269372, -0.0843926, 0.0843926
8: 0.8364359, 1.0181484, 0.8364359, 1.0181484, -0.1817126, 0.1817126
9: -0.0283033, 0.1083779, -0.0283033, 0.1083779, -0.1366812, 0.1366812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 3.70 + 597.87 = 601.57 seconds
