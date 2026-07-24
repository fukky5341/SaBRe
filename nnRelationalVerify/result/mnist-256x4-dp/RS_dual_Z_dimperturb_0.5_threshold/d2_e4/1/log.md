## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00710256


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726)
1: (-0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462)
2: (-0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314)
3: (-0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518)
4: (0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659)
5: (-0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140)
6: (0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490)
7: (0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0094056, 0.0094056)
8: (0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182)
9: (-0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.49 + 2.30 = 3.79 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0087355, upper bound: 0.0087355

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0077320, upper bound: 0.0077320
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0077320, upper bound: 0.0077321
time: 1.19 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.54 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.54
Output dim: 6, lower bound: -0.0077320, upper bound: 0.0077320
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.54
Output dim: 6, lower bound: -0.0077320, upper bound: 0.0077321

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0093981, 0.0093944
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0077304, upper bound: 0.0077303
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0077303, upper bound: 0.0077303
time: 1.13 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0093944, 0.0094056
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0077304, upper bound: 0.0077303
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0077303, upper bound: 0.0077303
time: 1.13 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.72 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.72
Output dim: 6, lower bound: -0.0077304, upper bound: 0.0077303
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.72
Output dim: 6, lower bound: -0.0077303, upper bound: 0.0077303
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.72
Output dim: 6, lower bound: -0.0077304, upper bound: 0.0077303
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.72
Output dim: 6, lower bound: -0.0077303, upper bound: 0.0077303

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0094558, 0.0094454
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0077093, upper bound: 0.0077064
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0077064, upper bound: 0.0077093
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0094495, 0.0094521
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0077093, upper bound: 0.0077066
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0077064, upper bound: 0.0077094
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0094521, 0.0094573
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0077093, upper bound: 0.0077064
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0077064, upper bound: 0.0077093
time: 1.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0094454, 0.0094640
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0077093, upper bound: 0.0077066
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0077064, upper bound: 0.0077094
time: 1.09 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.70 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 6, lower bound: -0.0077093, upper bound: 0.0077064
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 6, lower bound: -0.0077064, upper bound: 0.0077093
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 6, lower bound: -0.0077093, upper bound: 0.0077066
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 6, lower bound: -0.0077064, upper bound: 0.0077094
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 6, lower bound: -0.0077093, upper bound: 0.0077064
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 6, lower bound: -0.0077064, upper bound: 0.0077093
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 6, lower bound: -0.0077093, upper bound: 0.0077066
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 6, lower bound: -0.0077064, upper bound: 0.0077094

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0094604, 0.0094483
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074238, upper bound: 0.0074218
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074238, upper bound: 0.0074218
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0094579, 0.0094500
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074216, upper bound: 0.0074239
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074216, upper bound: 0.0074239
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0094541, 0.0094547
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074239, upper bound: 0.0074218
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074239, upper bound: 0.0074218
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0094521, 0.0094568
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074216, upper bound: 0.0074241
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074216, upper bound: 0.0074238
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0094568, 0.0094603
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074238, upper bound: 0.0074218
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074238, upper bound: 0.0074218
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0094547, 0.0094621
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074216, upper bound: 0.0074239
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074216, upper bound: 0.0074239
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0094500, 0.0094667
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074239, upper bound: 0.0074218
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074239, upper bound: 0.0074218
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0094483, 0.0094688
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074216, upper bound: 0.0074241
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074216, upper bound: 0.0074238
time: 1.04 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 5.49 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.49
Output dim: 6, lower bound: -0.0074238, upper bound: 0.0074218
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.49
Output dim: 6, lower bound: -0.0074238, upper bound: 0.0074218
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.49
Output dim: 6, lower bound: -0.0074216, upper bound: 0.0074239
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.49
Output dim: 6, lower bound: -0.0074216, upper bound: 0.0074239
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.49
Output dim: 6, lower bound: -0.0074239, upper bound: 0.0074218
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.49
Output dim: 6, lower bound: -0.0074239, upper bound: 0.0074218
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.49
Output dim: 6, lower bound: -0.0074216, upper bound: 0.0074241
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.49
Output dim: 6, lower bound: -0.0074216, upper bound: 0.0074238
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.49
Output dim: 6, lower bound: -0.0074238, upper bound: 0.0074218
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.49
Output dim: 6, lower bound: -0.0074238, upper bound: 0.0074218
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.49
Output dim: 6, lower bound: -0.0074216, upper bound: 0.0074239
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.49
Output dim: 6, lower bound: -0.0074216, upper bound: 0.0074239
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.49
Output dim: 6, lower bound: -0.0074239, upper bound: 0.0074218
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.49
Output dim: 6, lower bound: -0.0074239, upper bound: 0.0074218
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.49
Output dim: 6, lower bound: -0.0074216, upper bound: 0.0074241
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.49
Output dim: 6, lower bound: -0.0074216, upper bound: 0.0074238

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0094467, 0.0093756
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073120, upper bound: 0.0073078
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073095, upper bound: 0.0073097
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0093878, 0.0094378
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073120, upper bound: 0.0073078
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073095, upper bound: 0.0073097
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0094444, 0.0093774
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073096, upper bound: 0.0073096
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073080, upper bound: 0.0073120
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0093852, 0.0094393
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073096, upper bound: 0.0073096
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073080, upper bound: 0.0073120
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0094446, 0.0093820
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073120, upper bound: 0.0073080
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073096, upper bound: 0.0073096
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0093814, 0.0094387
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073120, upper bound: 0.0073080
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073096, upper bound: 0.0073096
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0094428, 0.0093841
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073096, upper bound: 0.0073095
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073078, upper bound: 0.0073120
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0093794, 0.0094408
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073096, upper bound: 0.0073095
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073078, upper bound: 0.0073120
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0094408, 0.0093876
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073120, upper bound: 0.0073078
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073095, upper bound: 0.0073097
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0093841, 0.0094498
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073120, upper bound: 0.0073078
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073095, upper bound: 0.0073097
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0094387, 0.0093894
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073096, upper bound: 0.0073096
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073080, upper bound: 0.0073120
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0093820, 0.0094513
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073096, upper bound: 0.0073096
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073080, upper bound: 0.0073120
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0094393, 0.0093940
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073120, upper bound: 0.0073080
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073096, upper bound: 0.0073096
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0093774, 0.0094507
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073120, upper bound: 0.0073080
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073096, upper bound: 0.0073096
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0094378, 0.0093961
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073096, upper bound: 0.0073095
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073078, upper bound: 0.0073120
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0093756, 0.0094528
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073096, upper bound: 0.0073095
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073078, upper bound: 0.0073120
time: 1.32 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 7.98 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.98
Output dim: 6, lower bound: -0.0073120, upper bound: 0.0073078
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.98
Output dim: 6, lower bound: -0.0073095, upper bound: 0.0073097
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.98
Output dim: 6, lower bound: -0.0073120, upper bound: 0.0073078
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.98
Output dim: 6, lower bound: -0.0073095, upper bound: 0.0073097
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.98
Output dim: 6, lower bound: -0.0073096, upper bound: 0.0073096
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.98
Output dim: 6, lower bound: -0.0073080, upper bound: 0.0073120
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.98
Output dim: 6, lower bound: -0.0073096, upper bound: 0.0073096
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.98
Output dim: 6, lower bound: -0.0073080, upper bound: 0.0073120
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.98
Output dim: 6, lower bound: -0.0073120, upper bound: 0.0073080
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.98
Output dim: 6, lower bound: -0.0073096, upper bound: 0.0073096
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.98
Output dim: 6, lower bound: -0.0073120, upper bound: 0.0073080
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.98
Output dim: 6, lower bound: -0.0073096, upper bound: 0.0073096
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.98
Output dim: 6, lower bound: -0.0073096, upper bound: 0.0073095
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.98
Output dim: 6, lower bound: -0.0073078, upper bound: 0.0073120
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.98
Output dim: 6, lower bound: -0.0073096, upper bound: 0.0073095
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.98
Output dim: 6, lower bound: -0.0073078, upper bound: 0.0073120
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.98
Output dim: 6, lower bound: -0.0073120, upper bound: 0.0073078
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.98
Output dim: 6, lower bound: -0.0073095, upper bound: 0.0073097
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.98
Output dim: 6, lower bound: -0.0073120, upper bound: 0.0073078
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.98
Output dim: 6, lower bound: -0.0073095, upper bound: 0.0073097
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.98
Output dim: 6, lower bound: -0.0073096, upper bound: 0.0073096
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.98
Output dim: 6, lower bound: -0.0073080, upper bound: 0.0073120
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.98
Output dim: 6, lower bound: -0.0073096, upper bound: 0.0073096
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.98
Output dim: 6, lower bound: -0.0073080, upper bound: 0.0073120
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.98
Output dim: 6, lower bound: -0.0073120, upper bound: 0.0073080
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.98
Output dim: 6, lower bound: -0.0073096, upper bound: 0.0073096
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.98
Output dim: 6, lower bound: -0.0073120, upper bound: 0.0073080
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.98
Output dim: 6, lower bound: -0.0073096, upper bound: 0.0073096
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.98
Output dim: 6, lower bound: -0.0073096, upper bound: 0.0073095
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.98
Output dim: 6, lower bound: -0.0073078, upper bound: 0.0073120
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.98
Output dim: 6, lower bound: -0.0073096, upper bound: 0.0073095
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.98
Output dim: 6, lower bound: -0.0073078, upper bound: 0.0073120

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0093583, 0.0092717
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069438, upper bound: 0.0069357
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069438, upper bound: 0.0069357
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0093428, 0.0092846
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069380, upper bound: 0.0069395
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069380, upper bound: 0.0069395
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0092958, 0.0093339
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069438, upper bound: 0.0069357
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069438, upper bound: 0.0069357
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0092838, 0.0093462
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069380, upper bound: 0.0069395
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069380, upper bound: 0.0069395
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0093557, 0.0092734
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069399, upper bound: 0.0069378
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069399, upper bound: 0.0069378
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0093405, 0.0092869
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069357, upper bound: 0.0069435
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069357, upper bound: 0.0069437
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0092933, 0.0093354
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069399, upper bound: 0.0069378
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069399, upper bound: 0.0069378
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0092813, 0.0093482
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069357, upper bound: 0.0069435
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069357, upper bound: 0.0069437
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0093548, 0.0092781
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069437, upper bound: 0.0069356
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069437, upper bound: 0.0069356
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0093407, 0.0092894
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069379, upper bound: 0.0069399
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069379, upper bound: 0.0069399
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0092913, 0.0093348
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069437, upper bound: 0.0069356
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069437, upper bound: 0.0069356
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0092774, 0.0093489
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069379, upper bound: 0.0069399
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069379, upper bound: 0.0069399
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0093526, 0.0092801
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069396, upper bound: 0.0069380
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069396, upper bound: 0.0069380
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0093389, 0.0092920
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069357, upper bound: 0.0069436
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069357, upper bound: 0.0069436
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0092891, 0.0093368
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069396, upper bound: 0.0069380
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069396, upper bound: 0.0069380
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0092754, 0.0093510
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069357, upper bound: 0.0069436
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069357, upper bound: 0.0069436
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0093510, 0.0092835
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069438, upper bound: 0.0069357
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069438, upper bound: 0.0069357
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0093368, 0.0092964
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069380, upper bound: 0.0069395
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069380, upper bound: 0.0069395
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0092920, 0.0093457
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069438, upper bound: 0.0069357
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069438, upper bound: 0.0069357
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0092801, 0.0093580
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069380, upper bound: 0.0069395
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069380, upper bound: 0.0069395
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0093489, 0.0092852
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069399, upper bound: 0.0069378
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069399, upper bound: 0.0069378
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0093348, 0.0092986
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069357, upper bound: 0.0069435
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069357, upper bound: 0.0069437
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0092894, 0.0093471
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069399, upper bound: 0.0069378
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069399, upper bound: 0.0069378
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0092781, 0.0093600
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069357, upper bound: 0.0069435
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069357, upper bound: 0.0069437
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0093482, 0.0092898
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069437, upper bound: 0.0069356
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069437, upper bound: 0.0069356
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0093354, 0.0093012
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069379, upper bound: 0.0069399
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069379, upper bound: 0.0069399
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0092869, 0.0093466
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069437, upper bound: 0.0069356
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069437, upper bound: 0.0069356
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0092734, 0.0093607
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069379, upper bound: 0.0069399
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069379, upper bound: 0.0069399
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0093463, 0.0092919
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069396, upper bound: 0.0069380
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069396, upper bound: 0.0069380
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0093339, 0.0093037
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069357, upper bound: 0.0069436
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069357, upper bound: 0.0069436
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0092846, 0.0093486
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069396, upper bound: 0.0069380
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069396, upper bound: 0.0069380
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037504, 0.0086229, 0.0037504, 0.0086229, -0.0048726, 0.0048726
1: -0.0002623, 0.0047839, -0.0002623, 0.0047839, -0.0050462, 0.0050462
2: -0.0258766, -0.0060452, -0.0258766, -0.0060452, -0.0198314, 0.0198314
3: -0.0019726, 0.0080792, -0.0019726, 0.0080792, -0.0100518, 0.0100518
4: 0.0117653, 0.0185311, 0.0117653, 0.0185311, -0.0067659, 0.0067659
5: -0.0032860, 0.0101280, -0.0032860, 0.0101280, -0.0134140, 0.0134140
6: 0.9944275, 1.0042765, 0.9944275, 1.0042765, -0.0098490, 0.0098490
7: 0.0079143, 0.0201617, 0.0079143, 0.0201617, -0.0092717, 0.0093628
8: 0.0017866, 0.0073049, 0.0017866, 0.0073049, -0.0055182, 0.0055182
9: -0.0276962, -0.0142505, -0.0276962, -0.0142505, -0.0134457, 0.0134457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069357, upper bound: 0.0069436
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069357, upper bound: 0.0069436
time: 0.96 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 9.38 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069438, upper bound: 0.0069357
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069438, upper bound: 0.0069357
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069380, upper bound: 0.0069395
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069380, upper bound: 0.0069395
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069438, upper bound: 0.0069357
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069438, upper bound: 0.0069357
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069380, upper bound: 0.0069395
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069380, upper bound: 0.0069395
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069399, upper bound: 0.0069378
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069399, upper bound: 0.0069378
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069357, upper bound: 0.0069435
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069357, upper bound: 0.0069437
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069399, upper bound: 0.0069378
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069399, upper bound: 0.0069378
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069357, upper bound: 0.0069435
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069357, upper bound: 0.0069437
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069437, upper bound: 0.0069356
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069437, upper bound: 0.0069356
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069379, upper bound: 0.0069399
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069379, upper bound: 0.0069399
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069437, upper bound: 0.0069356
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069437, upper bound: 0.0069356
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069379, upper bound: 0.0069399
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069379, upper bound: 0.0069399
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069396, upper bound: 0.0069380
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069396, upper bound: 0.0069380
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069357, upper bound: 0.0069436
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069357, upper bound: 0.0069436
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069396, upper bound: 0.0069380
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069396, upper bound: 0.0069380
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069357, upper bound: 0.0069436
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069357, upper bound: 0.0069436
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069438, upper bound: 0.0069357
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069438, upper bound: 0.0069357
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069380, upper bound: 0.0069395
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069380, upper bound: 0.0069395
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069438, upper bound: 0.0069357
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069438, upper bound: 0.0069357
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069380, upper bound: 0.0069395
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069380, upper bound: 0.0069395
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069399, upper bound: 0.0069378
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069399, upper bound: 0.0069378
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069357, upper bound: 0.0069435
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069357, upper bound: 0.0069437
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069399, upper bound: 0.0069378
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069399, upper bound: 0.0069378
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069357, upper bound: 0.0069435
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069357, upper bound: 0.0069437
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069437, upper bound: 0.0069356
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069437, upper bound: 0.0069356
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069379, upper bound: 0.0069399
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069379, upper bound: 0.0069399
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069437, upper bound: 0.0069356
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069437, upper bound: 0.0069356
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069379, upper bound: 0.0069399
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069379, upper bound: 0.0069399
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069396, upper bound: 0.0069380
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069396, upper bound: 0.0069380
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069357, upper bound: 0.0069436
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069357, upper bound: 0.0069436
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069396, upper bound: 0.0069380
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069396, upper bound: 0.0069380
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069357, upper bound: 0.0069436
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 9.38
Output dim: 6, lower bound: -0.0069357, upper bound: 0.0069436

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.79 + 496.76 = 500.54 seconds
