## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.06674455


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125)
1: (-0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677)
2: (-0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924)
3: (0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574)
4: (-0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169)
5: (-0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054)
6: (-0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380)
7: (-0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123)
8: (-0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561)
9: (-0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.21 + 3.08 = 4.29 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0785230, upper bound: 0.0785227

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 172

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0706520, upper bound: 0.0706525
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0706520, upper bound: 0.0706525
time: 1.58 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 3.16 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 3.16
Output dim: 3, lower bound: -0.0706520, upper bound: 0.0706525
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 3.16
Output dim: 3, lower bound: -0.0706520, upper bound: 0.0706525

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 154

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0593581, upper bound: 0.0593587
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0593581, upper bound: 0.0593587
time: 1.07 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0706524, upper bound: 0.0706525
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0706525, upper bound: 0.0706525
time: 1.98 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.56 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 4.56
Output dim: 3, lower bound: -0.0593581, upper bound: 0.0593587
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 4.56
Output dim: 3, lower bound: -0.0593581, upper bound: 0.0593587
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.56
Output dim: 3, lower bound: -0.0706524, upper bound: 0.0706525
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.56
Output dim: 3, lower bound: -0.0706525, upper bound: 0.0706525

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672123, upper bound: 0.0672128
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672123, upper bound: 0.0672128
time: 1.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0706522, upper bound: 0.0706515
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0706514, upper bound: 0.0706522
time: 2.28 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.77 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.77
Output dim: 3, lower bound: -0.0672123, upper bound: 0.0672128
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.77
Output dim: 3, lower bound: -0.0672123, upper bound: 0.0672128
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.77
Output dim: 3, lower bound: -0.0706522, upper bound: 0.0706515
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.77
Output dim: 3, lower bound: -0.0706514, upper bound: 0.0706522

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0511410, upper bound: 0.0511410
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0511410, upper bound: 0.0511407
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672120, upper bound: 0.0672117
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672117, upper bound: 0.0672119
time: 1.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0593578, upper bound: 0.0593581
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0593578, upper bound: 0.0593579
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672116, upper bound: 0.0672125
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672116, upper bound: 0.0672125
time: 1.20 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.49 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.49
Output dim: 3, lower bound: -0.0511410, upper bound: 0.0511410
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.49
Output dim: 3, lower bound: -0.0511410, upper bound: 0.0511407
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.49
Output dim: 3, lower bound: -0.0672120, upper bound: 0.0672117
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.49
Output dim: 3, lower bound: -0.0672117, upper bound: 0.0672119
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.49
Output dim: 3, lower bound: -0.0593578, upper bound: 0.0593581
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.49
Output dim: 3, lower bound: -0.0593578, upper bound: 0.0593579
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.49
Output dim: 3, lower bound: -0.0672116, upper bound: 0.0672125
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.49
Output dim: 3, lower bound: -0.0672116, upper bound: 0.0672125

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672118, upper bound: 0.0672115
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672119, upper bound: 0.0672117
time: 1.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0511401, upper bound: 0.0511405
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0511401, upper bound: 0.0511405
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671205, upper bound: 0.0671213
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671201, upper bound: 0.0671211
time: 1.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672116, upper bound: 0.0672121
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672115, upper bound: 0.0672122
time: 1.23 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.64 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.64
Output dim: 3, lower bound: -0.0672118, upper bound: 0.0672115
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.64
Output dim: 3, lower bound: -0.0672119, upper bound: 0.0672117
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 3, lower bound: -0.0511401, upper bound: 0.0511405
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 3, lower bound: -0.0511401, upper bound: 0.0511405
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.64
Output dim: 3, lower bound: -0.0671205, upper bound: 0.0671213
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.64
Output dim: 3, lower bound: -0.0671201, upper bound: 0.0671211
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.64
Output dim: 3, lower bound: -0.0672116, upper bound: 0.0672121
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.64
Output dim: 3, lower bound: -0.0672115, upper bound: 0.0672122

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671207, upper bound: 0.0671204
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671206, upper bound: 0.0671202
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671206, upper bound: 0.0671206
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671205, upper bound: 0.0671208
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510499, upper bound: 0.0510503
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510499, upper bound: 0.0510503
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671199, upper bound: 0.0671202
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671198, upper bound: 0.0671206
time: 1.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671203, upper bound: 0.0671210
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671203, upper bound: 0.0671208
time: 1.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671203, upper bound: 0.0671211
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671203, upper bound: 0.0671211
time: 1.47 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.06 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.06
Output dim: 3, lower bound: -0.0671207, upper bound: 0.0671204
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.06
Output dim: 3, lower bound: -0.0671206, upper bound: 0.0671202
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.06
Output dim: 3, lower bound: -0.0671206, upper bound: 0.0671206
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.06
Output dim: 3, lower bound: -0.0671205, upper bound: 0.0671208
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.06
Output dim: 3, lower bound: -0.0510499, upper bound: 0.0510503
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.06
Output dim: 3, lower bound: -0.0510499, upper bound: 0.0510503
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.06
Output dim: 3, lower bound: -0.0671199, upper bound: 0.0671202
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.06
Output dim: 3, lower bound: -0.0671198, upper bound: 0.0671206
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.06
Output dim: 3, lower bound: -0.0671203, upper bound: 0.0671210
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.06
Output dim: 3, lower bound: -0.0671203, upper bound: 0.0671208
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.06
Output dim: 3, lower bound: -0.0671203, upper bound: 0.0671211
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.06
Output dim: 3, lower bound: -0.0671203, upper bound: 0.0671211

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510498, upper bound: 0.0510498
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510498, upper bound: 0.0510498
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671205, upper bound: 0.0671199
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671200, upper bound: 0.0671202
time: 1.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671200, upper bound: 0.0671197
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671200, upper bound: 0.0671199
time: 1.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671200, upper bound: 0.0671198
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671200, upper bound: 0.0671202
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671202, upper bound: 0.0671201
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671201, upper bound: 0.0671198
time: 1.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671202, upper bound: 0.0671201
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671197, upper bound: 0.0671205
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671197, upper bound: 0.0671201
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671202, upper bound: 0.0671202
time: 1.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510497, upper bound: 0.0510500
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510497, upper bound: 0.0510500
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671201, upper bound: 0.0671205
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671197, upper bound: 0.0671205
time: 1.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671201, upper bound: 0.0671198
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671197, upper bound: 0.0671205
time: 1.21 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.91 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.91
Output dim: 3, lower bound: -0.0510498, upper bound: 0.0510498
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.91
Output dim: 3, lower bound: -0.0510498, upper bound: 0.0510498
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.91
Output dim: 3, lower bound: -0.0671205, upper bound: 0.0671199
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.91
Output dim: 3, lower bound: -0.0671200, upper bound: 0.0671202
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.91
Output dim: 3, lower bound: -0.0671200, upper bound: 0.0671197
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.91
Output dim: 3, lower bound: -0.0671200, upper bound: 0.0671199
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.91
Output dim: 3, lower bound: -0.0671200, upper bound: 0.0671198
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.91
Output dim: 3, lower bound: -0.0671200, upper bound: 0.0671202
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.91
Output dim: 3, lower bound: -0.0671202, upper bound: 0.0671201
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.91
Output dim: 3, lower bound: -0.0671201, upper bound: 0.0671198
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.91
Output dim: 3, lower bound: -0.0671202, upper bound: 0.0671201
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.91
Output dim: 3, lower bound: -0.0671197, upper bound: 0.0671205
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.91
Output dim: 3, lower bound: -0.0671197, upper bound: 0.0671201
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.91
Output dim: 3, lower bound: -0.0671202, upper bound: 0.0671202
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.91
Output dim: 3, lower bound: -0.0510497, upper bound: 0.0510500
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.91
Output dim: 3, lower bound: -0.0510497, upper bound: 0.0510500
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.91
Output dim: 3, lower bound: -0.0671201, upper bound: 0.0671205
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.91
Output dim: 3, lower bound: -0.0671197, upper bound: 0.0671205
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.91
Output dim: 3, lower bound: -0.0671201, upper bound: 0.0671198
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.91
Output dim: 3, lower bound: -0.0671197, upper bound: 0.0671205

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510495, upper bound: 0.0510495
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510495, upper bound: 0.0510495
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510495, upper bound: 0.0510495
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510495, upper bound: 0.0510495
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510494, upper bound: 0.0510495
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510494, upper bound: 0.0510495
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510495, upper bound: 0.0510495
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510495, upper bound: 0.0510495
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510494, upper bound: 0.0510495
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510494, upper bound: 0.0510495
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510495, upper bound: 0.0510495
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510495, upper bound: 0.0510495
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510493, upper bound: 0.0510497
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510493, upper bound: 0.0510497
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510493, upper bound: 0.0510497
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510493, upper bound: 0.0510497
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510493, upper bound: 0.0510497
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510493, upper bound: 0.0510497
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510493, upper bound: 0.0510497
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510493, upper bound: 0.0510497
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510492, upper bound: 0.0510494
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510492, upper bound: 0.0510494
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510492, upper bound: 0.0510497
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510492, upper bound: 0.0510497
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510492, upper bound: 0.0510497
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510492, upper bound: 0.0510497
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510492, upper bound: 0.0510497
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510492, upper bound: 0.0510497
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510493, upper bound: 0.0510497
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510493, upper bound: 0.0510497
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125
1: -0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677
2: -0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924
3: 0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574
4: -0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169
5: -0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054
6: -0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380
7: -0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123
8: -0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561
9: -0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510493, upper bound: 0.0510497
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0510493, upper bound: 0.0510497
time: 1.02 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 7.00 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 7.00
Output dim: 3, lower bound: -0.0510495, upper bound: 0.0510495
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 7.00
Output dim: 3, lower bound: -0.0510495, upper bound: 0.0510495
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 7.00
Output dim: 3, lower bound: -0.0510495, upper bound: 0.0510495
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 7.00
Output dim: 3, lower bound: -0.0510495, upper bound: 0.0510495
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 7.00
Output dim: 3, lower bound: -0.0510494, upper bound: 0.0510495
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 7.00
Output dim: 3, lower bound: -0.0510494, upper bound: 0.0510495
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 7.00
Output dim: 3, lower bound: -0.0510495, upper bound: 0.0510495
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 7.00
Output dim: 3, lower bound: -0.0510495, upper bound: 0.0510495
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 7.00
Output dim: 3, lower bound: -0.0510494, upper bound: 0.0510495
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 7.00
Output dim: 3, lower bound: -0.0510494, upper bound: 0.0510495
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 7.00
Output dim: 3, lower bound: -0.0510495, upper bound: 0.0510495
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 7.00
Output dim: 3, lower bound: -0.0510495, upper bound: 0.0510495
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 7.00
Output dim: 3, lower bound: -0.0510493, upper bound: 0.0510497
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 7.00
Output dim: 3, lower bound: -0.0510493, upper bound: 0.0510497
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 7.00
Output dim: 3, lower bound: -0.0510493, upper bound: 0.0510497
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 7.00
Output dim: 3, lower bound: -0.0510493, upper bound: 0.0510497
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 7.00
Output dim: 3, lower bound: -0.0510493, upper bound: 0.0510497
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 7.00
Output dim: 3, lower bound: -0.0510493, upper bound: 0.0510497
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 7.00
Output dim: 3, lower bound: -0.0510493, upper bound: 0.0510497
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 7.00
Output dim: 3, lower bound: -0.0510493, upper bound: 0.0510497
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 7.00
Output dim: 3, lower bound: -0.0510492, upper bound: 0.0510494
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 7.00
Output dim: 3, lower bound: -0.0510492, upper bound: 0.0510494
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 7.00
Output dim: 3, lower bound: -0.0510492, upper bound: 0.0510497
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 7.00
Output dim: 3, lower bound: -0.0510492, upper bound: 0.0510497
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 7.00
Output dim: 3, lower bound: -0.0510492, upper bound: 0.0510497
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 7.00
Output dim: 3, lower bound: -0.0510492, upper bound: 0.0510497
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 7.00
Output dim: 3, lower bound: -0.0510492, upper bound: 0.0510497
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 7.00
Output dim: 3, lower bound: -0.0510492, upper bound: 0.0510497
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 7.00
Output dim: 3, lower bound: -0.0510493, upper bound: 0.0510497
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 7.00
Output dim: 3, lower bound: -0.0510493, upper bound: 0.0510497
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 7.00
Output dim: 3, lower bound: -0.0510493, upper bound: 0.0510497
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 7.00
Output dim: 3, lower bound: -0.0510493, upper bound: 0.0510497

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 4.29 + 250.40 = 254.69 seconds
