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
execution time: IAR + RelationalAnalysis = 1.30 + 3.12 = 4.42 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0785230, upper bound: 0.0785227

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0740115, upper bound: 0.0740122
time: 2.19 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0740115, upper bound: 0.0740125
time: 2.20 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 4.51 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 4.51
Output dim: 3, lower bound: -0.0740115, upper bound: 0.0740122
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 4.51
Output dim: 3, lower bound: -0.0740115, upper bound: 0.0740125

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0740114, upper bound: 0.0740123
time: 2.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0740114, upper bound: 0.0740121
time: 1.86 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0740114, upper bound: 0.0740120
time: 2.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0740114, upper bound: 0.0740118
time: 1.71 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 5.00 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 5.00
Output dim: 3, lower bound: -0.0740114, upper bound: 0.0740123
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 5.00
Output dim: 3, lower bound: -0.0740114, upper bound: 0.0740121
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 5.00
Output dim: 3, lower bound: -0.0740114, upper bound: 0.0740120
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 5.00
Output dim: 3, lower bound: -0.0740114, upper bound: 0.0740118

## BFS RS instance: RS_RSZ1_RSZ1

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 172

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672121, upper bound: 0.0672126
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672121, upper bound: 0.0672126
time: 1.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 172

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672120, upper bound: 0.0672126
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672120, upper bound: 0.0672124
time: 1.54 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 172

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672121, upper bound: 0.0672124
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672121, upper bound: 0.0672126
time: 1.31 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 172

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672120, upper bound: 0.0672124
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672120, upper bound: 0.0672124
time: 1.42 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.09 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.09
Output dim: 3, lower bound: -0.0672121, upper bound: 0.0672126
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.09
Output dim: 3, lower bound: -0.0672121, upper bound: 0.0672126
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.09
Output dim: 3, lower bound: -0.0672120, upper bound: 0.0672126
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.09
Output dim: 3, lower bound: -0.0672120, upper bound: 0.0672124
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.09
Output dim: 3, lower bound: -0.0672121, upper bound: 0.0672124
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.09
Output dim: 3, lower bound: -0.0672121, upper bound: 0.0672126
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.09
Output dim: 3, lower bound: -0.0672120, upper bound: 0.0672124
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.09
Output dim: 3, lower bound: -0.0672120, upper bound: 0.0672124

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672121, upper bound: 0.0672121
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672121, upper bound: 0.0672123
time: 2.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672121, upper bound: 0.0672121
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672121, upper bound: 0.0672123
time: 2.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672125, upper bound: 0.0672121
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672120, upper bound: 0.0672125
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672125, upper bound: 0.0672124
time: 3.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672120, upper bound: 0.0672125
time: 1.19 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672121, upper bound: 0.0672124
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672121, upper bound: 0.0672125
time: 2.08 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672121, upper bound: 0.0672121
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672121, upper bound: 0.0672125
time: 2.10 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672125, upper bound: 0.0672121
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672120, upper bound: 0.0672126
time: 1.30 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672125, upper bound: 0.0672119
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672120, upper bound: 0.0672126
time: 1.30 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.96 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 3, lower bound: -0.0672121, upper bound: 0.0672121
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 3, lower bound: -0.0672121, upper bound: 0.0672123
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 3, lower bound: -0.0672121, upper bound: 0.0672121
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 3, lower bound: -0.0672121, upper bound: 0.0672123
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 3, lower bound: -0.0672125, upper bound: 0.0672121
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 3, lower bound: -0.0672120, upper bound: 0.0672125
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 3, lower bound: -0.0672125, upper bound: 0.0672124
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 3, lower bound: -0.0672120, upper bound: 0.0672125
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 3, lower bound: -0.0672121, upper bound: 0.0672124
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 3, lower bound: -0.0672121, upper bound: 0.0672125
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 3, lower bound: -0.0672121, upper bound: 0.0672121
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 3, lower bound: -0.0672121, upper bound: 0.0672125
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 3, lower bound: -0.0672125, upper bound: 0.0672121
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 3, lower bound: -0.0672120, upper bound: 0.0672126
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 3, lower bound: -0.0672125, upper bound: 0.0672119
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 3, lower bound: -0.0672120, upper bound: 0.0672126

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672118, upper bound: 0.0672117
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672115, upper bound: 0.0672119
time: 1.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672119, upper bound: 0.0672113
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672114, upper bound: 0.0672121
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672118, upper bound: 0.0672117
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672115, upper bound: 0.0672116
time: 1.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672119, upper bound: 0.0672113
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672116, upper bound: 0.0672119
time: 1.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672119, upper bound: 0.0672117
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672116, upper bound: 0.0672112
time: 1.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672119, upper bound: 0.0672114
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672115, upper bound: 0.0672120
time: 1.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672119, upper bound: 0.0672117
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672116, upper bound: 0.0672112
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672119, upper bound: 0.0672116
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672115, upper bound: 0.0672120
time: 1.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672118, upper bound: 0.0672115
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672115, upper bound: 0.0672122
time: 1.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672119, upper bound: 0.0672119
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672116, upper bound: 0.0672119
time: 1.51 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672118, upper bound: 0.0672115
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672115, upper bound: 0.0672122
time: 1.31 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672119, upper bound: 0.0672113
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672116, upper bound: 0.0672121
time: 1.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672119, upper bound: 0.0672117
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672116, upper bound: 0.0672112
time: 1.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672119, upper bound: 0.0672114
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672115, upper bound: 0.0672122
time: 1.22 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672119, upper bound: 0.0672117
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672116, upper bound: 0.0672112
time: 1.43 seconds

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
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672119, upper bound: 0.0672116
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672115, upper bound: 0.0672122
time: 1.21 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 7.77 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.77
Output dim: 3, lower bound: -0.0672118, upper bound: 0.0672117
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.77
Output dim: 3, lower bound: -0.0672115, upper bound: 0.0672119
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.77
Output dim: 3, lower bound: -0.0672119, upper bound: 0.0672113
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.77
Output dim: 3, lower bound: -0.0672114, upper bound: 0.0672121
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.77
Output dim: 3, lower bound: -0.0672118, upper bound: 0.0672117
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.77
Output dim: 3, lower bound: -0.0672115, upper bound: 0.0672116
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.77
Output dim: 3, lower bound: -0.0672119, upper bound: 0.0672113
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.77
Output dim: 3, lower bound: -0.0672116, upper bound: 0.0672119
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.77
Output dim: 3, lower bound: -0.0672119, upper bound: 0.0672117
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.77
Output dim: 3, lower bound: -0.0672116, upper bound: 0.0672112
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.77
Output dim: 3, lower bound: -0.0672119, upper bound: 0.0672114
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.77
Output dim: 3, lower bound: -0.0672115, upper bound: 0.0672120
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.77
Output dim: 3, lower bound: -0.0672119, upper bound: 0.0672117
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.77
Output dim: 3, lower bound: -0.0672116, upper bound: 0.0672112
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.77
Output dim: 3, lower bound: -0.0672119, upper bound: 0.0672116
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.77
Output dim: 3, lower bound: -0.0672115, upper bound: 0.0672120
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.77
Output dim: 3, lower bound: -0.0672118, upper bound: 0.0672115
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.77
Output dim: 3, lower bound: -0.0672115, upper bound: 0.0672122
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.77
Output dim: 3, lower bound: -0.0672119, upper bound: 0.0672119
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.77
Output dim: 3, lower bound: -0.0672116, upper bound: 0.0672119
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.77
Output dim: 3, lower bound: -0.0672118, upper bound: 0.0672115
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.77
Output dim: 3, lower bound: -0.0672115, upper bound: 0.0672122
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.77
Output dim: 3, lower bound: -0.0672119, upper bound: 0.0672113
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.77
Output dim: 3, lower bound: -0.0672116, upper bound: 0.0672121
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.77
Output dim: 3, lower bound: -0.0672119, upper bound: 0.0672117
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.77
Output dim: 3, lower bound: -0.0672116, upper bound: 0.0672112
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.77
Output dim: 3, lower bound: -0.0672119, upper bound: 0.0672114
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.77
Output dim: 3, lower bound: -0.0672115, upper bound: 0.0672122
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.77
Output dim: 3, lower bound: -0.0672119, upper bound: 0.0672117
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.77
Output dim: 3, lower bound: -0.0672116, upper bound: 0.0672112
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.77
Output dim: 3, lower bound: -0.0672119, upper bound: 0.0672116
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.77
Output dim: 3, lower bound: -0.0672115, upper bound: 0.0672122

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672109
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672110
time: 1.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672114
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672113, upper bound: 0.0672112
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672115, upper bound: 0.0672113
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672113
time: 1.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672113
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672108, upper bound: 0.0672114
time: 1.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672109
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672114
time: 1.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672116
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672113, upper bound: 0.0672112
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672109
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672113
time: 1.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672116
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672108, upper bound: 0.0672114
time: 1.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672109
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672113, upper bound: 0.0672114
time: 1.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672114
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672117
time: 2.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672111, upper bound: 0.0672113
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672111, upper bound: 0.0672109
time: 1.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672117
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672117
time: 1.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672113, upper bound: 0.0672113
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672114
time: 1.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672114
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672112
time: 1.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672111, upper bound: 0.0672113
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672111, upper bound: 0.0672113
time: 1.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672117
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672117
time: 1.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672113
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672109
time: 1.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672116
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672113, upper bound: 0.0672112
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672115, upper bound: 0.0672113
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672113
time: 1.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672113
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672108, upper bound: 0.0672114
time: 1.37 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672113
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672114
time: 1.28 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672114
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672113, upper bound: 0.0672112
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672113
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672113
time: 1.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672116
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672108, upper bound: 0.0672114
time: 1.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672109
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672113, upper bound: 0.0672104
time: 1.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672116
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672111
time: 1.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672111, upper bound: 0.0672114
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672114
time: 1.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672117
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672117
time: 1.59 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672113, upper bound: 0.0672110
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672114
time: 1.35 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672115
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672112
time: 1.38 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672111, upper bound: 0.0672114
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672114
time: 1.87 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672117
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672117
time: 1.60 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 7.75 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672109
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672110
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672114
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672113, upper bound: 0.0672112
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672115, upper bound: 0.0672113
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672113
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672113
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672108, upper bound: 0.0672114
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672109
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672114
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672116
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672113, upper bound: 0.0672112
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672109
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672113
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672116
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672108, upper bound: 0.0672114
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672109
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672113, upper bound: 0.0672114
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672114
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672117
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672111, upper bound: 0.0672113
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672111, upper bound: 0.0672109
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672117
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672117
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672113, upper bound: 0.0672113
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672114
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672114
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672112
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672111, upper bound: 0.0672113
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672111, upper bound: 0.0672113
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672117
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672117
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672113
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672109
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672116
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672113, upper bound: 0.0672112
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672115, upper bound: 0.0672113
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672113
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672113
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672108, upper bound: 0.0672114
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672113
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672114
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672114
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672113, upper bound: 0.0672112
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672113
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672113
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672116
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672108, upper bound: 0.0672114
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672109
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672113, upper bound: 0.0672104
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672116
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672111
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672111, upper bound: 0.0672114
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672114
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672117
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672117
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672113, upper bound: 0.0672110
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672114
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672115
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672112
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672111, upper bound: 0.0672114
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672112, upper bound: 0.0672114
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672117
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.75
Output dim: 3, lower bound: -0.0672109, upper bound: 0.0672117

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 154

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0511396, upper bound: 0.0511397
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0511396, upper bound: 0.0511397
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 154

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0511396, upper bound: 0.0511397
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0511396, upper bound: 0.0511397
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 154

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0511394, upper bound: 0.0511399
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0511394, upper bound: 0.0511399
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 154

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0511397, upper bound: 0.0511396
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0511397, upper bound: 0.0511396
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671205, upper bound: 0.0671202
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671200, upper bound: 0.0671194
time: 3.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671200, upper bound: 0.0671202
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671205, upper bound: 0.0671200
time: 1.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671197, upper bound: 0.0671199
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671201, upper bound: 0.0671203
time: 1.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671202, upper bound: 0.0671200
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671202, upper bound: 0.0671205
time: 1.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671200, upper bound: 0.0671200
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671205, upper bound: 0.0671196
time: 1.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671200, upper bound: 0.0671200
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671205, upper bound: 0.0671198
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671197, upper bound: 0.0671202
time: 10.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671202, upper bound: 0.0671200
time: 2.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671197, upper bound: 0.0671200
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671197, upper bound: 0.0671200
time: 1.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671205, upper bound: 0.0671202
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671200, upper bound: 0.0671195
time: 2.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 4.42 + 595.67 = 600.09 seconds
