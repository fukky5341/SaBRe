## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00164889


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0030697, 0.0051057, 0.0030697, 0.0051057, -0.0016650, 0.0016650)
1: (0.0017658, 0.0020599, 0.0017658, 0.0020599, -0.0002405, 0.0002405)
2: (0.0115371, 0.0126627, 0.0115371, 0.0126627, -0.0009205, 0.0009205)
3: (-0.0027483, -0.0015841, -0.0027483, -0.0015841, -0.0009521, 0.0009521)
4: (-0.0023221, -0.0010618, -0.0023221, -0.0010618, -0.0010307, 0.0010307)
5: (0.0051182, 0.0063108, 0.0051182, 0.0063108, -0.0009753, 0.0009753)
6: (-0.0019930, 0.0027392, -0.0019930, 0.0027392, -0.0038699, 0.0038699)
7: (-0.0062873, 0.0001575, -0.0062873, 0.0001575, -0.0052704, 0.0052704)
8: (0.9847850, 0.9893248, 0.9847850, 0.9893248, -0.0037126, 0.0037126)
9: (-0.0061971, -0.0020761, -0.0061971, -0.0020761, -0.0033701, 0.0033701)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.53 + 1.51 = 3.04 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0018224, upper bound: 0.0018224

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017036, upper bound: 0.0017581
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017581, upper bound: 0.0017036
time: 0.60 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.27 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.27
Output dim: 8, lower bound: -0.0017036, upper bound: 0.0017581
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.27
Output dim: 8, lower bound: -0.0017581, upper bound: 0.0017036

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0030697, 0.0051057, 0.0030697, 0.0051057, -0.0014162, 0.0014328
1: 0.0017658, 0.0020599, 0.0017658, 0.0020599, -0.0002046, 0.0002070
2: 0.0115371, 0.0126627, 0.0115371, 0.0126627, -0.0007922, 0.0007830
3: -0.0027483, -0.0015841, -0.0027483, -0.0015841, -0.0008193, 0.0008098
4: -0.0023221, -0.0010618, -0.0023221, -0.0010618, -0.0008767, 0.0008869
5: 0.0051182, 0.0063108, 0.0051182, 0.0063108, -0.0008393, 0.0008296
6: -0.0019930, 0.0027392, -0.0019930, 0.0027392, -0.0033302, 0.0032917
7: -0.0062873, 0.0001575, -0.0062873, 0.0001575, -0.0044830, 0.0045354
8: 0.9847850, 0.9893248, 0.9847850, 0.9893248, -0.0031579, 0.0031949
9: -0.0061971, -0.0020761, -0.0061971, -0.0020761, -0.0029001, 0.0028666

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016814, upper bound: 0.0017369
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016815, upper bound: 0.0017357
time: 0.65 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0030697, 0.0051057, 0.0030697, 0.0051057, -0.0014328, 0.0014162
1: 0.0017658, 0.0020599, 0.0017658, 0.0020599, -0.0002070, 0.0002046
2: 0.0115371, 0.0126627, 0.0115371, 0.0126627, -0.0007830, 0.0007922
3: -0.0027483, -0.0015841, -0.0027483, -0.0015841, -0.0008098, 0.0008193
4: -0.0023221, -0.0010618, -0.0023221, -0.0010618, -0.0008869, 0.0008767
5: 0.0051182, 0.0063108, 0.0051182, 0.0063108, -0.0008296, 0.0008393
6: -0.0019930, 0.0027392, -0.0019930, 0.0027392, -0.0032917, 0.0033302
7: -0.0062873, 0.0001575, -0.0062873, 0.0001575, -0.0045354, 0.0044830
8: 0.9847850, 0.9893248, 0.9847850, 0.9893248, -0.0031949, 0.0031579
9: -0.0061971, -0.0020761, -0.0061971, -0.0020761, -0.0028666, 0.0029001

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017417, upper bound: 0.0016864
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017420, upper bound: 0.0016867
time: 0.60 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.63 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.63
Output dim: 8, lower bound: -0.0016814, upper bound: 0.0017369
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.63
Output dim: 8, lower bound: -0.0016815, upper bound: 0.0017357
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.63
Output dim: 8, lower bound: -0.0017417, upper bound: 0.0016864
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.63
Output dim: 8, lower bound: -0.0017420, upper bound: 0.0016867

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0030697, 0.0051057, 0.0030697, 0.0051057, -0.0014156, 0.0014319
1: 0.0017658, 0.0020599, 0.0017658, 0.0020599, -0.0002045, 0.0002069
2: 0.0115371, 0.0126627, 0.0115371, 0.0126627, -0.0007917, 0.0007826
3: -0.0027483, -0.0015841, -0.0027483, -0.0015841, -0.0008188, 0.0008094
4: -0.0023221, -0.0010618, -0.0023221, -0.0010618, -0.0008763, 0.0008864
5: 0.0051182, 0.0063108, 0.0051182, 0.0063108, -0.0008388, 0.0008293
6: -0.0019930, 0.0027392, -0.0019930, 0.0027392, -0.0033282, 0.0032902
7: -0.0062873, 0.0001575, -0.0062873, 0.0001575, -0.0044810, 0.0045327
8: 0.9847850, 0.9893248, 0.9847850, 0.9893248, -0.0031565, 0.0031930
9: -0.0061971, -0.0020761, -0.0061971, -0.0020761, -0.0028984, 0.0028653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016646, upper bound: 0.0017212
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016642, upper bound: 0.0017207
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0030697, 0.0051057, 0.0030697, 0.0051057, -0.0014155, 0.0014322
1: 0.0017658, 0.0020599, 0.0017658, 0.0020599, -0.0002045, 0.0002069
2: 0.0115371, 0.0126627, 0.0115371, 0.0126627, -0.0007918, 0.0007826
3: -0.0027483, -0.0015841, -0.0027483, -0.0015841, -0.0008189, 0.0008094
4: -0.0023221, -0.0010618, -0.0023221, -0.0010618, -0.0008762, 0.0008865
5: 0.0051182, 0.0063108, 0.0051182, 0.0063108, -0.0008390, 0.0008292
6: -0.0019930, 0.0027392, -0.0019930, 0.0027392, -0.0033287, 0.0032900
7: -0.0062873, 0.0001575, -0.0062873, 0.0001575, -0.0044807, 0.0045334
8: 0.9847850, 0.9893248, 0.9847850, 0.9893248, -0.0031563, 0.0031934
9: -0.0061971, -0.0020761, -0.0061971, -0.0020761, -0.0028988, 0.0028651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016813, upper bound: 0.0017348
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016703, upper bound: 0.0017355
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0030697, 0.0051057, 0.0030697, 0.0051057, -0.0014280, 0.0014113
1: 0.0017658, 0.0020599, 0.0017658, 0.0020599, -0.0002063, 0.0002039
2: 0.0115371, 0.0126627, 0.0115371, 0.0126627, -0.0007803, 0.0007895
3: -0.0027483, -0.0015841, -0.0027483, -0.0015841, -0.0008070, 0.0008166
4: -0.0023221, -0.0010618, -0.0023221, -0.0010618, -0.0008840, 0.0008736
5: 0.0051182, 0.0063108, 0.0051182, 0.0063108, -0.0008268, 0.0008365
6: -0.0019930, 0.0027392, -0.0019930, 0.0027392, -0.0032803, 0.0033191
7: -0.0062873, 0.0001575, -0.0062873, 0.0001575, -0.0045204, 0.0044675
8: 0.9847850, 0.9893248, 0.9847850, 0.9893248, -0.0031842, 0.0031470
9: -0.0061971, -0.0020761, -0.0061971, -0.0020761, -0.0028567, 0.0028904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016416, upper bound: 0.0016178
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016794, upper bound: 0.0015915
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0030697, 0.0051057, 0.0030697, 0.0051057, -0.0014280, 0.0014115
1: 0.0017658, 0.0020599, 0.0017658, 0.0020599, -0.0002063, 0.0002039
2: 0.0115371, 0.0126627, 0.0115371, 0.0126627, -0.0007804, 0.0007895
3: -0.0027483, -0.0015841, -0.0027483, -0.0015841, -0.0008071, 0.0008165
4: -0.0023221, -0.0010618, -0.0023221, -0.0010618, -0.0008840, 0.0008737
5: 0.0051182, 0.0063108, 0.0051182, 0.0063108, -0.0008268, 0.0008365
6: -0.0019930, 0.0027392, -0.0019930, 0.0027392, -0.0032807, 0.0033191
7: -0.0062873, 0.0001575, -0.0062873, 0.0001575, -0.0045203, 0.0044680
8: 0.9847850, 0.9893248, 0.9847850, 0.9893248, -0.0031842, 0.0031473
9: -0.0061971, -0.0020761, -0.0061971, -0.0020761, -0.0028569, 0.0028904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016808, upper bound: 0.0016624
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017186, upper bound: 0.0016308
time: 0.60 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.67 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 8, lower bound: -0.0016646, upper bound: 0.0017212
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 8, lower bound: -0.0016642, upper bound: 0.0017207
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 8, lower bound: -0.0016813, upper bound: 0.0017348
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 8, lower bound: -0.0016703, upper bound: 0.0017355
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.67
Output dim: 8, lower bound: -0.0016416, upper bound: 0.0016178
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 8, lower bound: -0.0016794, upper bound: 0.0015915
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 8, lower bound: -0.0016808, upper bound: 0.0016624
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 8, lower bound: -0.0017186, upper bound: 0.0016308

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0030697, 0.0051057, 0.0030697, 0.0051057, -0.0014109, 0.0014274
1: 0.0017658, 0.0020599, 0.0017658, 0.0020599, -0.0002038, 0.0002062
2: 0.0115371, 0.0126627, 0.0115371, 0.0126627, -0.0007891, 0.0007801
3: -0.0027483, -0.0015841, -0.0027483, -0.0015841, -0.0008162, 0.0008068
4: -0.0023221, -0.0010618, -0.0023221, -0.0010618, -0.0008734, 0.0008836
5: 0.0051182, 0.0063108, 0.0051182, 0.0063108, -0.0008361, 0.0008265
6: -0.0019930, 0.0027392, -0.0019930, 0.0027392, -0.0033176, 0.0032794
7: -0.0062873, 0.0001575, -0.0062873, 0.0001575, -0.0044663, 0.0045182
8: 0.9847850, 0.9893248, 0.9847850, 0.9893248, -0.0031461, 0.0031827
9: -0.0061971, -0.0020761, -0.0061971, -0.0020761, -0.0028891, 0.0028559

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014413, upper bound: 0.0015065
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014413, upper bound: 0.0015065
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0030697, 0.0051057, 0.0030697, 0.0051057, -0.0014108, 0.0014273
1: 0.0017658, 0.0020599, 0.0017658, 0.0020599, -0.0002038, 0.0002062
2: 0.0115371, 0.0126627, 0.0115371, 0.0126627, -0.0007891, 0.0007800
3: -0.0027483, -0.0015841, -0.0027483, -0.0015841, -0.0008161, 0.0008067
4: -0.0023221, -0.0010618, -0.0023221, -0.0010618, -0.0008733, 0.0008835
5: 0.0051182, 0.0063108, 0.0051182, 0.0063108, -0.0008361, 0.0008264
6: -0.0019930, 0.0027392, -0.0019930, 0.0027392, -0.0033174, 0.0032791
7: -0.0062873, 0.0001575, -0.0062873, 0.0001575, -0.0044659, 0.0045180
8: 0.9847850, 0.9893248, 0.9847850, 0.9893248, -0.0031458, 0.0031826
9: -0.0061971, -0.0020761, -0.0061971, -0.0020761, -0.0028889, 0.0028556

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015691, upper bound: 0.0016563
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015950, upper bound: 0.0016216
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0030697, 0.0051057, 0.0030697, 0.0051057, -0.0014150, 0.0014304
1: 0.0017658, 0.0020599, 0.0017658, 0.0020599, -0.0002044, 0.0002066
2: 0.0115371, 0.0126627, 0.0115371, 0.0126627, -0.0007908, 0.0007823
3: -0.0027483, -0.0015841, -0.0027483, -0.0015841, -0.0008179, 0.0008091
4: -0.0023221, -0.0010618, -0.0023221, -0.0010618, -0.0008759, 0.0008854
5: 0.0051182, 0.0063108, 0.0051182, 0.0063108, -0.0008379, 0.0008289
6: -0.0019930, 0.0027392, -0.0019930, 0.0027392, -0.0033246, 0.0032889
7: -0.0062873, 0.0001575, -0.0062873, 0.0001575, -0.0044792, 0.0045278
8: 0.9847850, 0.9893248, 0.9847850, 0.9893248, -0.0031552, 0.0031895
9: -0.0061971, -0.0020761, -0.0061971, -0.0020761, -0.0028952, 0.0028641

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014574, upper bound: 0.0015182
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014574, upper bound: 0.0015182
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0030697, 0.0051057, 0.0030697, 0.0051057, -0.0014137, 0.0014315
1: 0.0017658, 0.0020599, 0.0017658, 0.0020599, -0.0002042, 0.0002068
2: 0.0115371, 0.0126627, 0.0115371, 0.0126627, -0.0007914, 0.0007816
3: -0.0027483, -0.0015841, -0.0027483, -0.0015841, -0.0008185, 0.0008084
4: -0.0023221, -0.0010618, -0.0023221, -0.0010618, -0.0008751, 0.0008861
5: 0.0051182, 0.0063108, 0.0051182, 0.0063108, -0.0008385, 0.0008281
6: -0.0019930, 0.0027392, -0.0019930, 0.0027392, -0.0033271, 0.0032858
7: -0.0062873, 0.0001575, -0.0062873, 0.0001575, -0.0044750, 0.0045312
8: 0.9847850, 0.9893248, 0.9847850, 0.9893248, -0.0031523, 0.0031919
9: -0.0061971, -0.0020761, -0.0061971, -0.0020761, -0.0028974, 0.0028614

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016117, upper bound: 0.0016777
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016117, upper bound: 0.0016774
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0030697, 0.0051057, 0.0030697, 0.0051057, -0.0013266, 0.0012923
1: 0.0017658, 0.0020599, 0.0017658, 0.0020599, -0.0001917, 0.0001867
2: 0.0115371, 0.0126627, 0.0115371, 0.0126627, -0.0007145, 0.0007335
3: -0.0027483, -0.0015841, -0.0027483, -0.0015841, -0.0007389, 0.0007586
4: -0.0023221, -0.0010618, -0.0023221, -0.0010618, -0.0008212, 0.0008000
5: 0.0051182, 0.0063108, 0.0051182, 0.0063108, -0.0007570, 0.0007771
6: -0.0019930, 0.0027392, -0.0019930, 0.0027392, -0.0030037, 0.0030835
7: -0.0062873, 0.0001575, -0.0062873, 0.0001575, -0.0041994, 0.0040907
8: 0.9847850, 0.9893248, 0.9847850, 0.9893248, -0.0029581, 0.0028816
9: -0.0061971, -0.0020761, -0.0061971, -0.0020761, -0.0026157, 0.0026852

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016792, upper bound: 0.0015838
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016790, upper bound: 0.0015913
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0030697, 0.0051057, 0.0030697, 0.0051057, -0.0014160, 0.0014022
1: 0.0017658, 0.0020599, 0.0017658, 0.0020599, -0.0002046, 0.0002026
2: 0.0115371, 0.0126627, 0.0115371, 0.0126627, -0.0007752, 0.0007829
3: -0.0027483, -0.0015841, -0.0027483, -0.0015841, -0.0008018, 0.0008097
4: -0.0023221, -0.0010618, -0.0023221, -0.0010618, -0.0008765, 0.0008680
5: 0.0051182, 0.0063108, 0.0051182, 0.0063108, -0.0008214, 0.0008295
6: -0.0019930, 0.0027392, -0.0019930, 0.0027392, -0.0032590, 0.0032912
7: -0.0062873, 0.0001575, -0.0062873, 0.0001575, -0.0044823, 0.0044385
8: 0.9847850, 0.9893248, 0.9847850, 0.9893248, -0.0031575, 0.0031266
9: -0.0061971, -0.0020761, -0.0061971, -0.0020761, -0.0028381, 0.0028661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016200, upper bound: 0.0016013
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016201, upper bound: 0.0016013
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0030697, 0.0051057, 0.0030697, 0.0051057, -0.0014186, 0.0013995
1: 0.0017658, 0.0020599, 0.0017658, 0.0020599, -0.0002050, 0.0002022
2: 0.0115371, 0.0126627, 0.0115371, 0.0126627, -0.0007737, 0.0007843
3: -0.0027483, -0.0015841, -0.0027483, -0.0015841, -0.0008002, 0.0008112
4: -0.0023221, -0.0010618, -0.0023221, -0.0010618, -0.0008782, 0.0008663
5: 0.0051182, 0.0063108, 0.0051182, 0.0063108, -0.0008198, 0.0008310
6: -0.0019930, 0.0027392, -0.0019930, 0.0027392, -0.0032528, 0.0032973
7: -0.0062873, 0.0001575, -0.0062873, 0.0001575, -0.0044907, 0.0044300
8: 0.9847850, 0.9893248, 0.9847850, 0.9893248, -0.0031633, 0.0031206
9: -0.0061971, -0.0020761, -0.0061971, -0.0020761, -0.0028327, 0.0028715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016961, upper bound: 0.0016089
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016981, upper bound: 0.0016089
time: 0.63 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.73 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.73
Output dim: 8, lower bound: -0.0014413, upper bound: 0.0015065
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.73
Output dim: 8, lower bound: -0.0014413, upper bound: 0.0015065
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 8, lower bound: -0.0015691, upper bound: 0.0016563
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.73
Output dim: 8, lower bound: -0.0015950, upper bound: 0.0016216
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.73
Output dim: 8, lower bound: -0.0014574, upper bound: 0.0015182
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.73
Output dim: 8, lower bound: -0.0014574, upper bound: 0.0015182
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 8, lower bound: -0.0016117, upper bound: 0.0016777
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 8, lower bound: -0.0016117, upper bound: 0.0016774
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 8, lower bound: -0.0016792, upper bound: 0.0015838
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 8, lower bound: -0.0016790, upper bound: 0.0015913
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.73
Output dim: 8, lower bound: -0.0016200, upper bound: 0.0016013
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.73
Output dim: 8, lower bound: -0.0016201, upper bound: 0.0016013
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 8, lower bound: -0.0016961, upper bound: 0.0016089
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 8, lower bound: -0.0016981, upper bound: 0.0016089

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0030697, 0.0051057, 0.0030697, 0.0051057, -0.0012918, 0.0013260
1: 0.0017658, 0.0020599, 0.0017658, 0.0020599, -0.0001866, 0.0001916
2: 0.0115371, 0.0126627, 0.0115371, 0.0126627, -0.0007331, 0.0007142
3: -0.0027483, -0.0015841, -0.0027483, -0.0015841, -0.0007582, 0.0007387
4: -0.0023221, -0.0010618, -0.0023221, -0.0010618, -0.0007996, 0.0008208
5: 0.0051182, 0.0063108, 0.0051182, 0.0063108, -0.0007768, 0.0007567
6: -0.0019930, 0.0027392, -0.0019930, 0.0027392, -0.0030820, 0.0030025
7: -0.0062873, 0.0001575, -0.0062873, 0.0001575, -0.0040891, 0.0041974
8: 0.9847850, 0.9893248, 0.9847850, 0.9893248, -0.0028804, 0.0029568
9: -0.0061971, -0.0020761, -0.0061971, -0.0020761, -0.0026840, 0.0026147

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015170, upper bound: 0.0016323
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015449, upper bound: 0.0015890
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0030697, 0.0051057, 0.0030697, 0.0051057, -0.0013681, 0.0013859
1: 0.0017658, 0.0020599, 0.0017658, 0.0020599, -0.0001977, 0.0002002
2: 0.0115371, 0.0126627, 0.0115371, 0.0126627, -0.0007662, 0.0007564
3: -0.0027483, -0.0015841, -0.0027483, -0.0015841, -0.0007925, 0.0007823
4: -0.0023221, -0.0010618, -0.0023221, -0.0010618, -0.0008469, 0.0008579
5: 0.0051182, 0.0063108, 0.0051182, 0.0063108, -0.0008118, 0.0008015
6: -0.0019930, 0.0027392, -0.0019930, 0.0027392, -0.0032211, 0.0031799
7: -0.0062873, 0.0001575, -0.0062873, 0.0001575, -0.0043308, 0.0043869
8: 0.9847850, 0.9893248, 0.9847850, 0.9893248, -0.0030507, 0.0030902
9: -0.0061971, -0.0020761, -0.0061971, -0.0020761, -0.0028051, 0.0027692

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015512, upper bound: 0.0016529
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015875, upper bound: 0.0016151
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0030697, 0.0051057, 0.0030697, 0.0051057, -0.0013676, 0.0013859
1: 0.0017658, 0.0020599, 0.0017658, 0.0020599, -0.0001976, 0.0002002
2: 0.0115371, 0.0126627, 0.0115371, 0.0126627, -0.0007662, 0.0007561
3: -0.0027483, -0.0015841, -0.0027483, -0.0015841, -0.0007925, 0.0007820
4: -0.0023221, -0.0010618, -0.0023221, -0.0010618, -0.0008466, 0.0008579
5: 0.0051182, 0.0063108, 0.0051182, 0.0063108, -0.0008119, 0.0008011
6: -0.0019930, 0.0027392, -0.0019930, 0.0027392, -0.0032212, 0.0031787
7: -0.0062873, 0.0001575, -0.0062873, 0.0001575, -0.0043291, 0.0043870
8: 0.9847850, 0.9893248, 0.9847850, 0.9893248, -0.0030495, 0.0030903
9: -0.0061971, -0.0020761, -0.0061971, -0.0020761, -0.0028052, 0.0027682

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015211, upper bound: 0.0016151
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015454, upper bound: 0.0015759
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0030697, 0.0051057, 0.0030697, 0.0051057, -0.0013251, 0.0012903
1: 0.0017658, 0.0020599, 0.0017658, 0.0020599, -0.0001914, 0.0001864
2: 0.0115371, 0.0126627, 0.0115371, 0.0126627, -0.0007134, 0.0007326
3: -0.0027483, -0.0015841, -0.0027483, -0.0015841, -0.0007378, 0.0007577
4: -0.0023221, -0.0010618, -0.0023221, -0.0010618, -0.0008203, 0.0007987
5: 0.0051182, 0.0063108, 0.0051182, 0.0063108, -0.0007559, 0.0007763
6: -0.0019930, 0.0027392, -0.0019930, 0.0027392, -0.0029990, 0.0030800
7: -0.0062873, 0.0001575, -0.0062873, 0.0001575, -0.0041947, 0.0040844
8: 0.9847850, 0.9893248, 0.9847850, 0.9893248, -0.0029548, 0.0028771
9: -0.0061971, -0.0020761, -0.0061971, -0.0020761, -0.0026117, 0.0026822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016288, upper bound: 0.0015359
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016290, upper bound: 0.0015345
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0030697, 0.0051057, 0.0030697, 0.0051057, -0.0013246, 0.0012916
1: 0.0017658, 0.0020599, 0.0017658, 0.0020599, -0.0001914, 0.0001866
2: 0.0115371, 0.0126627, 0.0115371, 0.0126627, -0.0007141, 0.0007324
3: -0.0027483, -0.0015841, -0.0027483, -0.0015841, -0.0007385, 0.0007574
4: -0.0023221, -0.0010618, -0.0023221, -0.0010618, -0.0008200, 0.0007995
5: 0.0051182, 0.0063108, 0.0051182, 0.0063108, -0.0007566, 0.0007760
6: -0.0019930, 0.0027392, -0.0019930, 0.0027392, -0.0030020, 0.0030788
7: -0.0062873, 0.0001575, -0.0062873, 0.0001575, -0.0041931, 0.0040884
8: 0.9847850, 0.9893248, 0.9847850, 0.9893248, -0.0029537, 0.0028800
9: -0.0061971, -0.0020761, -0.0061971, -0.0020761, -0.0026143, 0.0026812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016560, upper bound: 0.0015699
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016560, upper bound: 0.0015690
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0030697, 0.0051057, 0.0030697, 0.0051057, -0.0014181, 0.0013988
1: 0.0017658, 0.0020599, 0.0017658, 0.0020599, -0.0002049, 0.0002021
2: 0.0115371, 0.0126627, 0.0115371, 0.0126627, -0.0007734, 0.0007840
3: -0.0027483, -0.0015841, -0.0027483, -0.0015841, -0.0007999, 0.0008109
4: -0.0023221, -0.0010618, -0.0023221, -0.0010618, -0.0008778, 0.0008659
5: 0.0051182, 0.0063108, 0.0051182, 0.0063108, -0.0008194, 0.0008307
6: -0.0019930, 0.0027392, -0.0019930, 0.0027392, -0.0032513, 0.0032961
7: -0.0062873, 0.0001575, -0.0062873, 0.0001575, -0.0044890, 0.0044280
8: 0.9847850, 0.9893248, 0.9847850, 0.9893248, -0.0031621, 0.0031192
9: -0.0061971, -0.0020761, -0.0061971, -0.0020761, -0.0028314, 0.0028704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015944, upper bound: 0.0015422
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016322, upper bound: 0.0015168
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0030697, 0.0051057, 0.0030697, 0.0051057, -0.0014181, 0.0013989
1: 0.0017658, 0.0020599, 0.0017658, 0.0020599, -0.0002049, 0.0002021
2: 0.0115371, 0.0126627, 0.0115371, 0.0126627, -0.0007734, 0.0007840
3: -0.0027483, -0.0015841, -0.0027483, -0.0015841, -0.0007999, 0.0008109
4: -0.0023221, -0.0010618, -0.0023221, -0.0010618, -0.0008778, 0.0008660
5: 0.0051182, 0.0063108, 0.0051182, 0.0063108, -0.0008195, 0.0008307
6: -0.0019930, 0.0027392, -0.0019930, 0.0027392, -0.0032515, 0.0032959
7: -0.0062873, 0.0001575, -0.0062873, 0.0001575, -0.0044888, 0.0044283
8: 0.9847850, 0.9893248, 0.9847850, 0.9893248, -0.0031620, 0.0031194
9: -0.0061971, -0.0020761, -0.0061971, -0.0020761, -0.0028316, 0.0028703

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016383, upper bound: 0.0015494
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016383, upper bound: 0.0015490
time: 0.61 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.69 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0015170, upper bound: 0.0016323
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0015449, upper bound: 0.0015890
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0015512, upper bound: 0.0016529
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0015875, upper bound: 0.0016151
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0015211, upper bound: 0.0016151
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0015454, upper bound: 0.0015759
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0016288, upper bound: 0.0015359
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0016290, upper bound: 0.0015345
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0016560, upper bound: 0.0015699
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0016560, upper bound: 0.0015690
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0015944, upper bound: 0.0015422
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0016322, upper bound: 0.0015168
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0016383, upper bound: 0.0015494
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0016383, upper bound: 0.0015490

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0030697, 0.0051057, 0.0030697, 0.0051057, -0.0013560, 0.0013776
1: 0.0017658, 0.0020599, 0.0017658, 0.0020599, -0.0001959, 0.0001990
2: 0.0115371, 0.0126627, 0.0115371, 0.0126627, -0.0007616, 0.0007497
3: -0.0027483, -0.0015841, -0.0027483, -0.0015841, -0.0007877, 0.0007754
4: -0.0023221, -0.0010618, -0.0023221, -0.0010618, -0.0008394, 0.0008528
5: 0.0051182, 0.0063108, 0.0051182, 0.0063108, -0.0008070, 0.0007943
6: -0.0019930, 0.0027392, -0.0019930, 0.0027392, -0.0032019, 0.0031516
7: -0.0062873, 0.0001575, -0.0062873, 0.0001575, -0.0042922, 0.0043607
8: 0.9847850, 0.9893248, 0.9847850, 0.9893248, -0.0030235, 0.0030718
9: -0.0061971, -0.0020761, -0.0061971, -0.0020761, -0.0027884, 0.0027446

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0013109, upper bound: 0.0015851
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014669, upper bound: 0.0013907
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0030697, 0.0051057, 0.0030697, 0.0051057, -0.0013239, 0.0012907
1: 0.0017658, 0.0020599, 0.0017658, 0.0020599, -0.0001913, 0.0001865
2: 0.0115371, 0.0126627, 0.0115371, 0.0126627, -0.0007136, 0.0007319
3: -0.0027483, -0.0015841, -0.0027483, -0.0015841, -0.0007380, 0.0007570
4: -0.0023221, -0.0010618, -0.0023221, -0.0010618, -0.0008195, 0.0007990
5: 0.0051182, 0.0063108, 0.0051182, 0.0063108, -0.0007561, 0.0007755
6: -0.0019930, 0.0027392, -0.0019930, 0.0027392, -0.0030000, 0.0030770
7: -0.0062873, 0.0001575, -0.0062873, 0.0001575, -0.0041906, 0.0040857
8: 0.9847850, 0.9893248, 0.9847850, 0.9893248, -0.0029520, 0.0028781
9: -0.0061971, -0.0020761, -0.0061971, -0.0020761, -0.0026125, 0.0026796

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016070, upper bound: 0.0015222
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016077, upper bound: 0.0015216
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0030697, 0.0051057, 0.0030697, 0.0051057, -0.0013238, 0.0012908
1: 0.0017658, 0.0020599, 0.0017658, 0.0020599, -0.0001912, 0.0001865
2: 0.0115371, 0.0126627, 0.0115371, 0.0126627, -0.0007137, 0.0007319
3: -0.0027483, -0.0015841, -0.0027483, -0.0015841, -0.0007381, 0.0007569
4: -0.0023221, -0.0010618, -0.0023221, -0.0010618, -0.0008194, 0.0007990
5: 0.0051182, 0.0063108, 0.0051182, 0.0063108, -0.0007562, 0.0007755
6: -0.0019930, 0.0027392, -0.0019930, 0.0027392, -0.0030002, 0.0030768
7: -0.0062873, 0.0001575, -0.0062873, 0.0001575, -0.0041903, 0.0040860
8: 0.9847850, 0.9893248, 0.9847850, 0.9893248, -0.0029518, 0.0028783
9: -0.0061971, -0.0020761, -0.0061971, -0.0020761, -0.0026127, 0.0026794

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016428, upper bound: 0.0015557
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016434, upper bound: 0.0015561
time: 0.73 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.82 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 8, lower bound: -0.0013109, upper bound: 0.0015851
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 8, lower bound: -0.0014669, upper bound: 0.0013907
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 8, lower bound: -0.0016070, upper bound: 0.0015222
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 8, lower bound: -0.0016077, upper bound: 0.0015216
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 8, lower bound: -0.0016428, upper bound: 0.0015557
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 8, lower bound: -0.0016434, upper bound: 0.0015561

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.04 + 63.51 = 66.55 seconds
