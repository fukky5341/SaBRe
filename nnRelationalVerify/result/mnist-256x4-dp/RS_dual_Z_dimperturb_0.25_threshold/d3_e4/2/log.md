## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0024309


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040211, -0.0036452, -0.0040211, -0.0036452, -0.0002695, 0.0002695)
1: (-0.0000683, 0.0020130, -0.0000683, 0.0020130, -0.0014925, 0.0014925)
2: (0.0104688, 0.0151188, 0.0104688, 0.0151188, -0.0033344, 0.0033344)
3: (0.0009632, 0.0029228, 0.0009632, 0.0029228, -0.0014051, 0.0014051)
4: (1.0004873, 1.0080895, 1.0004873, 1.0080895, -0.0054513, 0.0054513)
5: (0.0023362, 0.0038151, 0.0023362, 0.0038151, -0.0010605, 0.0010605)
6: (-0.0107078, -0.0087832, -0.0107078, -0.0087832, -0.0013801, 0.0013801)
7: (-0.0101692, -0.0099237, -0.0101692, -0.0099237, -0.0001760, 0.0001760)
8: (-0.0047856, -0.0034559, -0.0047856, -0.0034559, -0.0009535, 0.0009535)
9: (-0.0008699, 0.0057871, -0.0008699, 0.0057871, -0.0047735, 0.0047735)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.86 + 1.66 = 3.52 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0037024, upper bound: 0.0037025

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0033736, upper bound: 0.0034858
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0034858, upper bound: 0.0033737
time: 0.75 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.55 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.55
Output dim: 4, lower bound: -0.0033736, upper bound: 0.0034858
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.55
Output dim: 4, lower bound: -0.0034858, upper bound: 0.0033737

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040211, -0.0036452, -0.0040211, -0.0036452, -0.0002639, 0.0002565
1: -0.0000683, 0.0020130, -0.0000683, 0.0020130, -0.0014202, 0.0014612
2: 0.0104688, 0.0151188, 0.0104688, 0.0151188, -0.0032645, 0.0031729
3: 0.0009632, 0.0029228, 0.0009632, 0.0029228, -0.0013371, 0.0013757
4: 1.0004873, 1.0080895, 1.0004873, 1.0080895, -0.0051873, 0.0053371
5: 0.0023362, 0.0038151, 0.0023362, 0.0038151, -0.0010091, 0.0010383
6: -0.0107078, -0.0087832, -0.0107078, -0.0087832, -0.0013512, 0.0013132
7: -0.0101692, -0.0099237, -0.0101692, -0.0099237, -0.0001724, 0.0001675
8: -0.0047856, -0.0034559, -0.0047856, -0.0034559, -0.0009073, 0.0009336
9: -0.0008699, 0.0057871, -0.0008699, 0.0057871, -0.0046736, 0.0045424

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0032765, upper bound: 0.0033946
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0032778, upper bound: 0.0033886
time: 0.75 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040211, -0.0036452, -0.0040211, -0.0036452, -0.0002565, 0.0002695
1: -0.0000683, 0.0020130, -0.0000683, 0.0020130, -0.0014925, 0.0014202
2: 0.0104688, 0.0151188, 0.0104688, 0.0151188, -0.0031729, 0.0033344
3: 0.0009632, 0.0029228, 0.0009632, 0.0029228, -0.0014051, 0.0013371
4: 1.0004873, 1.0080895, 1.0004873, 1.0080895, -0.0054513, 0.0051873
5: 0.0023362, 0.0038151, 0.0023362, 0.0038151, -0.0010605, 0.0010091
6: -0.0107078, -0.0087832, -0.0107078, -0.0087832, -0.0013132, 0.0013801
7: -0.0101692, -0.0099237, -0.0101692, -0.0099237, -0.0001675, 0.0001760
8: -0.0047856, -0.0034559, -0.0047856, -0.0034559, -0.0009535, 0.0009073
9: -0.0008699, 0.0057871, -0.0008699, 0.0057871, -0.0045424, 0.0047735

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0033886, upper bound: 0.0032778
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0033945, upper bound: 0.0032765
time: 0.67 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.28 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 4, lower bound: -0.0032765, upper bound: 0.0033946
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 4, lower bound: -0.0032778, upper bound: 0.0033886
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 4, lower bound: -0.0033886, upper bound: 0.0032778
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 4, lower bound: -0.0033945, upper bound: 0.0032765

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040211, -0.0036452, -0.0040211, -0.0036452, -0.0002477, 0.0002411
1: -0.0000683, 0.0020130, -0.0000683, 0.0020130, -0.0013348, 0.0013716
2: 0.0104688, 0.0151188, 0.0104688, 0.0151188, -0.0030643, 0.0029821
3: 0.0009632, 0.0029228, 0.0009632, 0.0029228, -0.0012567, 0.0012913
4: 1.0004873, 1.0080895, 1.0004873, 1.0080895, -0.0048754, 0.0050097
5: 0.0023362, 0.0038151, 0.0023362, 0.0038151, -0.0009485, 0.0009746
6: -0.0107078, -0.0087832, -0.0107078, -0.0087832, -0.0012683, 0.0012343
7: -0.0101692, -0.0099237, -0.0101692, -0.0099237, -0.0001618, 0.0001574
8: -0.0047856, -0.0034559, -0.0047856, -0.0034559, -0.0008528, 0.0008763
9: -0.0008699, 0.0057871, -0.0008699, 0.0057871, -0.0043869, 0.0042693

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0032477, upper bound: 0.0033535
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0032373, upper bound: 0.0033619
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040211, -0.0036452, -0.0040211, -0.0036452, -0.0002485, 0.0002400
1: -0.0000683, 0.0020130, -0.0000683, 0.0020130, -0.0013289, 0.0013759
2: 0.0104688, 0.0151188, 0.0104688, 0.0151188, -0.0030738, 0.0029688
3: 0.0009632, 0.0029228, 0.0009632, 0.0029228, -0.0012511, 0.0012953
4: 1.0004873, 1.0080895, 1.0004873, 1.0080895, -0.0048537, 0.0050253
5: 0.0023362, 0.0038151, 0.0023362, 0.0038151, -0.0009442, 0.0009776
6: -0.0107078, -0.0087832, -0.0107078, -0.0087832, -0.0012722, 0.0012288
7: -0.0101692, -0.0099237, -0.0101692, -0.0099237, -0.0001623, 0.0001567
8: -0.0047856, -0.0034559, -0.0047856, -0.0034559, -0.0008490, 0.0008790
9: -0.0008699, 0.0057871, -0.0008699, 0.0057871, -0.0044005, 0.0042502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0032487, upper bound: 0.0033482
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0032390, upper bound: 0.0033563
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040211, -0.0036452, -0.0040211, -0.0036452, -0.0002400, 0.0002541
1: -0.0000683, 0.0020130, -0.0000683, 0.0020130, -0.0014069, 0.0013289
2: 0.0104688, 0.0151188, 0.0104688, 0.0151188, -0.0029688, 0.0031433
3: 0.0009632, 0.0029228, 0.0009632, 0.0029228, -0.0013246, 0.0012511
4: 1.0004873, 1.0080895, 1.0004873, 1.0080895, -0.0051389, 0.0048537
5: 0.0023362, 0.0038151, 0.0023362, 0.0038151, -0.0009997, 0.0009442
6: -0.0107078, -0.0087832, -0.0107078, -0.0087832, -0.0012288, 0.0013010
7: -0.0101692, -0.0099237, -0.0101692, -0.0099237, -0.0001567, 0.0001660
8: -0.0047856, -0.0034559, -0.0047856, -0.0034559, -0.0008989, 0.0008490
9: -0.0008699, 0.0057871, -0.0008699, 0.0057871, -0.0042502, 0.0045000

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0033563, upper bound: 0.0032390
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0033482, upper bound: 0.0032487
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040211, -0.0036452, -0.0040211, -0.0036452, -0.0002411, 0.0002530
1: -0.0000683, 0.0020130, -0.0000683, 0.0020130, -0.0014010, 0.0013348
2: 0.0104688, 0.0151188, 0.0104688, 0.0151188, -0.0029821, 0.0031300
3: 0.0009632, 0.0029228, 0.0009632, 0.0029228, -0.0013190, 0.0012567
4: 1.0004873, 1.0080895, 1.0004873, 1.0080895, -0.0051171, 0.0048754
5: 0.0023362, 0.0038151, 0.0023362, 0.0038151, -0.0009955, 0.0009485
6: -0.0107078, -0.0087832, -0.0107078, -0.0087832, -0.0012343, 0.0012955
7: -0.0101692, -0.0099237, -0.0101692, -0.0099237, -0.0001574, 0.0001652
8: -0.0047856, -0.0034559, -0.0047856, -0.0034559, -0.0008951, 0.0008528
9: -0.0008699, 0.0057871, -0.0008699, 0.0057871, -0.0042693, 0.0044809

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0033619, upper bound: 0.0032373
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0033535, upper bound: 0.0032477
time: 0.67 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.16 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 4, lower bound: -0.0032477, upper bound: 0.0033535
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 4, lower bound: -0.0032373, upper bound: 0.0033619
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 4, lower bound: -0.0032487, upper bound: 0.0033482
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 4, lower bound: -0.0032390, upper bound: 0.0033563
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 4, lower bound: -0.0033563, upper bound: 0.0032390
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 4, lower bound: -0.0033482, upper bound: 0.0032487
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 4, lower bound: -0.0033619, upper bound: 0.0032373
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 4, lower bound: -0.0033535, upper bound: 0.0032477

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040211, -0.0036452, -0.0040211, -0.0036452, -0.0002428, 0.0002366
1: -0.0000683, 0.0020130, -0.0000683, 0.0020130, -0.0013098, 0.0013445
2: 0.0104688, 0.0151188, 0.0104688, 0.0151188, -0.0030038, 0.0029263
3: 0.0009632, 0.0029228, 0.0009632, 0.0029228, -0.0012332, 0.0012658
4: 1.0004873, 1.0080895, 1.0004873, 1.0080895, -0.0047842, 0.0049108
5: 0.0023362, 0.0038151, 0.0023362, 0.0038151, -0.0009307, 0.0009553
6: -0.0107078, -0.0087832, -0.0107078, -0.0087832, -0.0012432, 0.0012112
7: -0.0101692, -0.0099237, -0.0101692, -0.0099237, -0.0001586, 0.0001545
8: -0.0047856, -0.0034559, -0.0047856, -0.0034559, -0.0008368, 0.0008590
9: -0.0008699, 0.0057871, -0.0008699, 0.0057871, -0.0043003, 0.0041894

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0021306, upper bound: 0.0021278
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0021306, upper bound: 0.0021278
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040211, -0.0036452, -0.0040211, -0.0036452, -0.0002431, 0.0002362
1: -0.0000683, 0.0020130, -0.0000683, 0.0020130, -0.0013077, 0.0013458
2: 0.0104688, 0.0151188, 0.0104688, 0.0151188, -0.0030067, 0.0029216
3: 0.0009632, 0.0029228, 0.0009632, 0.0029228, -0.0012312, 0.0012670
4: 1.0004873, 1.0080895, 1.0004873, 1.0080895, -0.0047765, 0.0049156
5: 0.0023362, 0.0038151, 0.0023362, 0.0038151, -0.0009292, 0.0009563
6: -0.0107078, -0.0087832, -0.0107078, -0.0087832, -0.0012444, 0.0012093
7: -0.0101692, -0.0099237, -0.0101692, -0.0099237, -0.0001587, 0.0001543
8: -0.0047856, -0.0034559, -0.0047856, -0.0034559, -0.0008355, 0.0008598
9: -0.0008699, 0.0057871, -0.0008699, 0.0057871, -0.0043044, 0.0041827

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0021249, upper bound: 0.0021372
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0021249, upper bound: 0.0021372
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040211, -0.0036452, -0.0040211, -0.0036452, -0.0002436, 0.0002354
1: -0.0000683, 0.0020130, -0.0000683, 0.0020130, -0.0013032, 0.0013488
2: 0.0104688, 0.0151188, 0.0104688, 0.0151188, -0.0030133, 0.0029116
3: 0.0009632, 0.0029228, 0.0009632, 0.0029228, -0.0012269, 0.0012698
4: 1.0004873, 1.0080895, 1.0004873, 1.0080895, -0.0047600, 0.0049264
5: 0.0023362, 0.0038151, 0.0023362, 0.0038151, -0.0009260, 0.0009584
6: -0.0107078, -0.0087832, -0.0107078, -0.0087832, -0.0012472, 0.0012051
7: -0.0101692, -0.0099237, -0.0101692, -0.0099237, -0.0001591, 0.0001537
8: -0.0047856, -0.0034559, -0.0047856, -0.0034559, -0.0008326, 0.0008617
9: -0.0008699, 0.0057871, -0.0008699, 0.0057871, -0.0043139, 0.0041683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0021372, upper bound: 0.0021249
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0021372, upper bound: 0.0021249
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040211, -0.0036452, -0.0040211, -0.0036452, -0.0002440, 0.0002351
1: -0.0000683, 0.0020130, -0.0000683, 0.0020130, -0.0013018, 0.0013509
2: 0.0104688, 0.0151188, 0.0104688, 0.0151188, -0.0030180, 0.0029083
3: 0.0009632, 0.0029228, 0.0009632, 0.0029228, -0.0012256, 0.0012718
4: 1.0004873, 1.0080895, 1.0004873, 1.0080895, -0.0047548, 0.0049340
5: 0.0023362, 0.0038151, 0.0023362, 0.0038151, -0.0009250, 0.0009599
6: -0.0107078, -0.0087832, -0.0107078, -0.0087832, -0.0012491, 0.0012037
7: -0.0101692, -0.0099237, -0.0101692, -0.0099237, -0.0001593, 0.0001535
8: -0.0047856, -0.0034559, -0.0047856, -0.0034559, -0.0008317, 0.0008630
9: -0.0008699, 0.0057871, -0.0008699, 0.0057871, -0.0043206, 0.0041636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0021278, upper bound: 0.0021306
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0021278, upper bound: 0.0021306
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040211, -0.0036452, -0.0040211, -0.0036452, -0.0002351, 0.0002496
1: -0.0000683, 0.0020130, -0.0000683, 0.0020130, -0.0013820, 0.0013018
2: 0.0104688, 0.0151188, 0.0104688, 0.0151188, -0.0029083, 0.0030875
3: 0.0009632, 0.0029228, 0.0009632, 0.0029228, -0.0013011, 0.0012256
4: 1.0004873, 1.0080895, 1.0004873, 1.0080895, -0.0050476, 0.0047548
5: 0.0023362, 0.0038151, 0.0023362, 0.0038151, -0.0009820, 0.0009250
6: -0.0107078, -0.0087832, -0.0107078, -0.0087832, -0.0012037, 0.0012779
7: -0.0101692, -0.0099237, -0.0101692, -0.0099237, -0.0001535, 0.0001630
8: -0.0047856, -0.0034559, -0.0047856, -0.0034559, -0.0008829, 0.0008317
9: -0.0008699, 0.0057871, -0.0008699, 0.0057871, -0.0041636, 0.0044201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0021306, upper bound: 0.0021278
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0021306, upper bound: 0.0021278
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040211, -0.0036452, -0.0040211, -0.0036452, -0.0002354, 0.0002492
1: -0.0000683, 0.0020130, -0.0000683, 0.0020130, -0.0013799, 0.0013032
2: 0.0104688, 0.0151188, 0.0104688, 0.0151188, -0.0029116, 0.0030828
3: 0.0009632, 0.0029228, 0.0009632, 0.0029228, -0.0012991, 0.0012269
4: 1.0004873, 1.0080895, 1.0004873, 1.0080895, -0.0050400, 0.0047600
5: 0.0023362, 0.0038151, 0.0023362, 0.0038151, -0.0009805, 0.0009260
6: -0.0107078, -0.0087832, -0.0107078, -0.0087832, -0.0012051, 0.0012760
7: -0.0101692, -0.0099237, -0.0101692, -0.0099237, -0.0001537, 0.0001628
8: -0.0047856, -0.0034559, -0.0047856, -0.0034559, -0.0008816, 0.0008326
9: -0.0008699, 0.0057871, -0.0008699, 0.0057871, -0.0041683, 0.0044134

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0021249, upper bound: 0.0021372
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0021249, upper bound: 0.0021372
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040211, -0.0036452, -0.0040211, -0.0036452, -0.0002362, 0.0002484
1: -0.0000683, 0.0020130, -0.0000683, 0.0020130, -0.0013754, 0.0013077
2: 0.0104688, 0.0151188, 0.0104688, 0.0151188, -0.0029216, 0.0030727
3: 0.0009632, 0.0029228, 0.0009632, 0.0029228, -0.0012948, 0.0012312
4: 1.0004873, 1.0080895, 1.0004873, 1.0080895, -0.0050235, 0.0047765
5: 0.0023362, 0.0038151, 0.0023362, 0.0038151, -0.0009773, 0.0009292
6: -0.0107078, -0.0087832, -0.0107078, -0.0087832, -0.0012093, 0.0012718
7: -0.0101692, -0.0099237, -0.0101692, -0.0099237, -0.0001543, 0.0001622
8: -0.0047856, -0.0034559, -0.0047856, -0.0034559, -0.0008787, 0.0008355
9: -0.0008699, 0.0057871, -0.0008699, 0.0057871, -0.0041827, 0.0043990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0021372, upper bound: 0.0021249
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0021372, upper bound: 0.0021249
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040211, -0.0036452, -0.0040211, -0.0036452, -0.0002366, 0.0002481
1: -0.0000683, 0.0020130, -0.0000683, 0.0020130, -0.0013739, 0.0013098
2: 0.0104688, 0.0151188, 0.0104688, 0.0151188, -0.0029263, 0.0030695
3: 0.0009632, 0.0029228, 0.0009632, 0.0029228, -0.0012935, 0.0012332
4: 1.0004873, 1.0080895, 1.0004873, 1.0080895, -0.0050182, 0.0047842
5: 0.0023362, 0.0038151, 0.0023362, 0.0038151, -0.0009762, 0.0009307
6: -0.0107078, -0.0087832, -0.0107078, -0.0087832, -0.0012112, 0.0012704
7: -0.0101692, -0.0099237, -0.0101692, -0.0099237, -0.0001545, 0.0001621
8: -0.0047856, -0.0034559, -0.0047856, -0.0034559, -0.0008778, 0.0008368
9: -0.0008699, 0.0057871, -0.0008699, 0.0057871, -0.0041894, 0.0043944

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0021278, upper bound: 0.0021306
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0021278, upper bound: 0.0021306
time: 0.56 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.97 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.97
Output dim: 4, lower bound: -0.0021306, upper bound: 0.0021278
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.97
Output dim: 4, lower bound: -0.0021306, upper bound: 0.0021278
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.97
Output dim: 4, lower bound: -0.0021249, upper bound: 0.0021372
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.97
Output dim: 4, lower bound: -0.0021249, upper bound: 0.0021372
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.97
Output dim: 4, lower bound: -0.0021372, upper bound: 0.0021249
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.97
Output dim: 4, lower bound: -0.0021372, upper bound: 0.0021249
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.97
Output dim: 4, lower bound: -0.0021278, upper bound: 0.0021306
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.97
Output dim: 4, lower bound: -0.0021278, upper bound: 0.0021306
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.97
Output dim: 4, lower bound: -0.0021306, upper bound: 0.0021278
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.97
Output dim: 4, lower bound: -0.0021306, upper bound: 0.0021278
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.97
Output dim: 4, lower bound: -0.0021249, upper bound: 0.0021372
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.97
Output dim: 4, lower bound: -0.0021249, upper bound: 0.0021372
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.97
Output dim: 4, lower bound: -0.0021372, upper bound: 0.0021249
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.97
Output dim: 4, lower bound: -0.0021372, upper bound: 0.0021249
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.97
Output dim: 4, lower bound: -0.0021278, upper bound: 0.0021306
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.97
Output dim: 4, lower bound: -0.0021278, upper bound: 0.0021306

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.52 + 61.98 = 65.50 seconds
