## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00035568


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0028027, 0.0028027)
1: (-0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004971, 0.0004971)
2: (0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0035214, 0.0035214)
3: (1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008822, 0.0008822)
4: (-0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005381, 0.0005381)
5: (0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0021393, 0.0021393)
6: (-0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001386, 0.0001386)
7: (-0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0052140, 0.0052140)
8: (-0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0055891, 0.0055891)
9: (-0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0026380, 0.0026380)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.33 + 1.77 = 3.10 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0004715, upper bound: 0.0004715

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004686, upper bound: 0.0004667
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004667, upper bound: 0.0004686
time: 1.00 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.16 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.16
Output dim: 3, lower bound: -0.0004686, upper bound: 0.0004667
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.16
Output dim: 3, lower bound: -0.0004667, upper bound: 0.0004686

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027989, 0.0027988
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004961, 0.0004962
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0035180, 0.0035180
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008781, 0.0008774
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005379, 0.0005379
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0021366, 0.0021365
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001377, 0.0001377
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0051982, 0.0051986
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0055886, 0.0055885
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0026372, 0.0026372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004592, upper bound: 0.0004552
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004571, upper bound: 0.0004576
time: 1.01 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027988, 0.0027989
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004962, 0.0004961
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0035180, 0.0035180
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008774, 0.0008781
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005379, 0.0005379
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0021365, 0.0021366
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001377, 0.0001377
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0051986, 0.0051982
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0055885, 0.0055886
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0026372, 0.0026372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004576, upper bound: 0.0004571
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004553, upper bound: 0.0004592
time: 1.10 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.45 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.45
Output dim: 3, lower bound: -0.0004592, upper bound: 0.0004552
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.45
Output dim: 3, lower bound: -0.0004571, upper bound: 0.0004576
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.45
Output dim: 3, lower bound: -0.0004576, upper bound: 0.0004571
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.45
Output dim: 3, lower bound: -0.0004553, upper bound: 0.0004592

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027600, 0.0027486
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004920, 0.0004912
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034708, 0.0034581
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008735, 0.0008727
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005291, 0.0005308
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0021071, 0.0020984
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001353, 0.0001345
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0050859, 0.0051134
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0055014, 0.0055174
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0026053, 0.0025985

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004560, upper bound: 0.0004511
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004540, upper bound: 0.0004521
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027486, 0.0027599
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004911, 0.0004921
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034581, 0.0034708
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008735, 0.0008727
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005308, 0.0005291
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020985, 0.0021070
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001345, 0.0001353
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0051126, 0.0050863
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0055174, 0.0055013
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025985, 0.0026054

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004539, upper bound: 0.0004530
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004524, upper bound: 0.0004544
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027599, 0.0027486
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004921, 0.0004911
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034708, 0.0034581
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008727, 0.0008735
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005291, 0.0005308
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0021070, 0.0020985
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001353, 0.0001345
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0050863, 0.0051126
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0055013, 0.0055174
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0026054, 0.0025985

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004544, upper bound: 0.0004524
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004529, upper bound: 0.0004539
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027486, 0.0027600
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004912, 0.0004920
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034581, 0.0034708
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008727, 0.0008735
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005308, 0.0005291
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020984, 0.0021071
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001345, 0.0001353
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0051134, 0.0050859
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0055174, 0.0055014
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025985, 0.0026053

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004520, upper bound: 0.0004540
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004511, upper bound: 0.0004560
time: 0.97 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.36 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 3, lower bound: -0.0004560, upper bound: 0.0004511
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 3, lower bound: -0.0004540, upper bound: 0.0004521
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 3, lower bound: -0.0004539, upper bound: 0.0004530
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 3, lower bound: -0.0004524, upper bound: 0.0004544
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 3, lower bound: -0.0004544, upper bound: 0.0004524
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 3, lower bound: -0.0004529, upper bound: 0.0004539
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 3, lower bound: -0.0004520, upper bound: 0.0004540
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 3, lower bound: -0.0004511, upper bound: 0.0004560

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027506, 0.0027408
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004910, 0.0004913
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034600, 0.0034497
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008738, 0.0008708
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005279, 0.0005291
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020999, 0.0020926
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001353, 0.0001345
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0050705, 0.0050993
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054896, 0.0055003
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025977, 0.0025943

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004436, upper bound: 0.0004428
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004476, upper bound: 0.0004382
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027522, 0.0027386
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004920, 0.0004903
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034624, 0.0034457
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008716, 0.0008730
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005271, 0.0005295
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0021012, 0.0020909
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001353, 0.0001345
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0050698, 0.0050980
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054808, 0.0055056
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0026011, 0.0025898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004420, upper bound: 0.0004438
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004456, upper bound: 0.0004393
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027387, 0.0027521
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004902, 0.0004921
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034457, 0.0034624
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008737, 0.0008707
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005295, 0.0005271
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020909, 0.0021012
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001345, 0.0001353
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0050972, 0.0050700
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0055057, 0.0054807
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025898, 0.0026012

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004422, upper bound: 0.0004446
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004456, upper bound: 0.0004404
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027409, 0.0027505
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004911, 0.0004911
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034497, 0.0034600
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008716, 0.0008730
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005291, 0.0005279
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020927, 0.0020999
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001345, 0.0001353
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0050985, 0.0050709
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0055004, 0.0054896
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025943, 0.0025978

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004406, upper bound: 0.0004460
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004441, upper bound: 0.0004415
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027505, 0.0027409
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004911, 0.0004911
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034600, 0.0034497
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008730, 0.0008716
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005279, 0.0005291
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020999, 0.0020927
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001353, 0.0001345
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0050709, 0.0050985
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054896, 0.0055004
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025978, 0.0025943

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004415, upper bound: 0.0004442
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004460, upper bound: 0.0004406
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027521, 0.0027387
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004921, 0.0004902
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034624, 0.0034457
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008707, 0.0008737
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005271, 0.0005295
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0021012, 0.0020909
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001353, 0.0001345
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0050700, 0.0050972
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054807, 0.0055057
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0026012, 0.0025898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004404, upper bound: 0.0004456
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004446, upper bound: 0.0004422
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027386, 0.0027522
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004903, 0.0004920
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034457, 0.0034624
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008730, 0.0008716
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005295, 0.0005271
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020909, 0.0021012
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001345, 0.0001353
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0050980, 0.0050698
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0055056, 0.0054808
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025898, 0.0026011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004392, upper bound: 0.0004456
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004438, upper bound: 0.0004420
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027408, 0.0027506
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004913, 0.0004910
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034497, 0.0034600
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008708, 0.0008738
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005291, 0.0005279
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020926, 0.0020999
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001345, 0.0001353
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0050993, 0.0050705
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0055003, 0.0054896
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025943, 0.0025977

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004382, upper bound: 0.0004476
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004428, upper bound: 0.0004437
time: 1.00 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.64 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 3, lower bound: -0.0004436, upper bound: 0.0004428
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 3, lower bound: -0.0004476, upper bound: 0.0004382
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 3, lower bound: -0.0004420, upper bound: 0.0004438
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 3, lower bound: -0.0004456, upper bound: 0.0004393
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 3, lower bound: -0.0004422, upper bound: 0.0004446
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 3, lower bound: -0.0004456, upper bound: 0.0004404
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 3, lower bound: -0.0004406, upper bound: 0.0004460
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 3, lower bound: -0.0004441, upper bound: 0.0004415
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 3, lower bound: -0.0004415, upper bound: 0.0004442
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 3, lower bound: -0.0004460, upper bound: 0.0004406
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 3, lower bound: -0.0004404, upper bound: 0.0004456
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 3, lower bound: -0.0004446, upper bound: 0.0004422
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 3, lower bound: -0.0004392, upper bound: 0.0004456
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 3, lower bound: -0.0004438, upper bound: 0.0004420
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 3, lower bound: -0.0004382, upper bound: 0.0004476
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 3, lower bound: -0.0004428, upper bound: 0.0004437

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027517, 0.0027446
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004868, 0.0004875
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034575, 0.0034510
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008347, 0.0008306
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005276, 0.0005282
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0021004, 0.0020952
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001325, 0.0001318
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0050812, 0.0051075
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054840, 0.0054875
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025890, 0.0025889

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003606, upper bound: 0.0003607
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003606, upper bound: 0.0003607
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027544, 0.0027425
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004872, 0.0004871
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034613, 0.0034484
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008341, 0.0008317
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005271, 0.0005288
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0021025, 0.0020936
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001327, 0.0001317
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0050794, 0.0051099
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054787, 0.0054947
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025923, 0.0025860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003628, upper bound: 0.0003584
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003628, upper bound: 0.0003584
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027532, 0.0027424
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004875, 0.0004865
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034599, 0.0034470
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008324, 0.0008331
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005268, 0.0005287
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0021016, 0.0020934
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001325, 0.0001318
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0050804, 0.0051062
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054751, 0.0054933
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025921, 0.0025844

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003584, upper bound: 0.0003628
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003584, upper bound: 0.0003628
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027561, 0.0027405
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004882, 0.0004862
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034637, 0.0034446
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008316, 0.0008339
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005265, 0.0005293
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0021038, 0.0020919
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001327, 0.0001317
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0050788, 0.0051087
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054709, 0.0055000
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025957, 0.0025821

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003607, upper bound: 0.0003606
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003607, upper bound: 0.0003606
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027406, 0.0027560
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004861, 0.0004883
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034446, 0.0034637
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008346, 0.0008307
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005293, 0.0005265
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020920, 0.0021037
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001318, 0.0001327
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0051078, 0.0050792
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0055000, 0.0054709
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025820, 0.0025957

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003606, upper bound: 0.0003607
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003606, upper bound: 0.0003607
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027425, 0.0027533
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004864, 0.0004876
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034470, 0.0034601
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008340, 0.0008316
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005287, 0.0005268
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020935, 0.0021017
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001319, 0.0001325
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0051054, 0.0050806
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054933, 0.0054751
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025843, 0.0025922

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003628, upper bound: 0.0003584
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003628, upper bound: 0.0003584
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027425, 0.0027544
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004870, 0.0004873
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034484, 0.0034613
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008324, 0.0008332
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005289, 0.0005271
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020936, 0.0021025
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001318, 0.0001327
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0051091, 0.0050798
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054947, 0.0054787
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025859, 0.0025924

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003584, upper bound: 0.0003628
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003584, upper bound: 0.0003628
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027447, 0.0027517
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004874, 0.0004869
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034510, 0.0034575
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008315, 0.0008338
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005282, 0.0005276
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020952, 0.0021004
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001319, 0.0001325
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0051071, 0.0050815
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054876, 0.0054840
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025888, 0.0025890

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003607, upper bound: 0.0003606
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003607, upper bound: 0.0003606
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027517, 0.0027447
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004869, 0.0004874
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034575, 0.0034510
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008338, 0.0008315
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005276, 0.0005282
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0021004, 0.0020952
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001325, 0.0001319
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0050815, 0.0051071
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054840, 0.0054876
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025890, 0.0025888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003606, upper bound: 0.0003607
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003606, upper bound: 0.0003607
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027544, 0.0027425
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004873, 0.0004870
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034613, 0.0034484
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008332, 0.0008324
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005271, 0.0005289
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0021025, 0.0020936
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001327, 0.0001318
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0050798, 0.0051091
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054787, 0.0054947
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025924, 0.0025859

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003628, upper bound: 0.0003584
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003628, upper bound: 0.0003584
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027533, 0.0027425
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004876, 0.0004864
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034601, 0.0034470
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008316, 0.0008340
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005268, 0.0005287
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0021017, 0.0020935
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001325, 0.0001319
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0050806, 0.0051054
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054751, 0.0054933
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025922, 0.0025843

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003584, upper bound: 0.0003628
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003584, upper bound: 0.0003628
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027560, 0.0027406
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004883, 0.0004861
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034637, 0.0034446
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008307, 0.0008346
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005265, 0.0005293
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0021037, 0.0020920
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001327, 0.0001318
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0050792, 0.0051078
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054709, 0.0055000
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025957, 0.0025820

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003607, upper bound: 0.0003606
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003607, upper bound: 0.0003606
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027405, 0.0027561
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004862, 0.0004882
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034446, 0.0034637
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008339, 0.0008316
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005293, 0.0005265
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020919, 0.0021038
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001317, 0.0001327
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0051087, 0.0050788
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0055000, 0.0054709
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025821, 0.0025957

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003606, upper bound: 0.0003607
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003606, upper bound: 0.0003607
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027424, 0.0027532
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004865, 0.0004875
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034470, 0.0034599
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008331, 0.0008324
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005287, 0.0005268
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020934, 0.0021016
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001318, 0.0001325
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0051062, 0.0050804
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054933, 0.0054751
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025844, 0.0025921

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003628, upper bound: 0.0003584
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003628, upper bound: 0.0003584
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027425, 0.0027544
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004871, 0.0004872
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034484, 0.0034613
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008317, 0.0008341
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005288, 0.0005271
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020936, 0.0021025
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001317, 0.0001327
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0051099, 0.0050794
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054947, 0.0054787
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025860, 0.0025923

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003584, upper bound: 0.0003628
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003584, upper bound: 0.0003628
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027446, 0.0027517
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004875, 0.0004868
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034510, 0.0034575
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008306, 0.0008347
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005282, 0.0005276
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020952, 0.0021004
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001318, 0.0001325
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0051075, 0.0050812
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054875, 0.0054840
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025889, 0.0025890

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003607, upper bound: 0.0003606
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003607, upper bound: 0.0003606
time: 1.05 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.55 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 3, lower bound: -0.0003606, upper bound: 0.0003607
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 3, lower bound: -0.0003606, upper bound: 0.0003607
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 3, lower bound: -0.0003628, upper bound: 0.0003584
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 3, lower bound: -0.0003628, upper bound: 0.0003584
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 3, lower bound: -0.0003584, upper bound: 0.0003628
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 3, lower bound: -0.0003584, upper bound: 0.0003628
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 3, lower bound: -0.0003607, upper bound: 0.0003606
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 3, lower bound: -0.0003607, upper bound: 0.0003606
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 3, lower bound: -0.0003606, upper bound: 0.0003607
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 3, lower bound: -0.0003606, upper bound: 0.0003607
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 3, lower bound: -0.0003628, upper bound: 0.0003584
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 3, lower bound: -0.0003628, upper bound: 0.0003584
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 3, lower bound: -0.0003584, upper bound: 0.0003628
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 3, lower bound: -0.0003584, upper bound: 0.0003628
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 3, lower bound: -0.0003607, upper bound: 0.0003606
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 3, lower bound: -0.0003607, upper bound: 0.0003606
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 3, lower bound: -0.0003606, upper bound: 0.0003607
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 3, lower bound: -0.0003606, upper bound: 0.0003607
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 3, lower bound: -0.0003628, upper bound: 0.0003584
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 3, lower bound: -0.0003628, upper bound: 0.0003584
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 3, lower bound: -0.0003584, upper bound: 0.0003628
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 3, lower bound: -0.0003584, upper bound: 0.0003628
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 3, lower bound: -0.0003607, upper bound: 0.0003606
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 3, lower bound: -0.0003607, upper bound: 0.0003606
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 3, lower bound: -0.0003606, upper bound: 0.0003607
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 3, lower bound: -0.0003606, upper bound: 0.0003607
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 3, lower bound: -0.0003628, upper bound: 0.0003584
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 3, lower bound: -0.0003628, upper bound: 0.0003584
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 3, lower bound: -0.0003584, upper bound: 0.0003628
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 3, lower bound: -0.0003584, upper bound: 0.0003628
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 3, lower bound: -0.0003607, upper bound: 0.0003606
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 3, lower bound: -0.0003607, upper bound: 0.0003606

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027494, 0.0027428
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004861, 0.0004868
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034542, 0.0034484
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008331, 0.0008300
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005272, 0.0005276
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020986, 0.0020937
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001324, 0.0001318
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0050799, 0.0051047
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054786, 0.0054816
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025862, 0.0025862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003413, upper bound: 0.0003469
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003472, upper bound: 0.0003410
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027499, 0.0027446
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004861, 0.0004875
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034549, 0.0034510
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008347, 0.0008290
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005276, 0.0005277
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020990, 0.0020952
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001325, 0.0001317
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0050812, 0.0051062
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054840, 0.0054822
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025862, 0.0025889

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003413, upper bound: 0.0003469
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003472, upper bound: 0.0003410
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027517, 0.0027406
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004865, 0.0004864
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034574, 0.0034458
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008325, 0.0008309
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005267, 0.0005281
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0021004, 0.0020922
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001326, 0.0001317
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0050781, 0.0051067
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054733, 0.0054863
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025885, 0.0025833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003429, upper bound: 0.0003448
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003490, upper bound: 0.0003393
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027526, 0.0027425
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004865, 0.0004871
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034587, 0.0034484
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008341, 0.0008301
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005271, 0.0005284
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0021011, 0.0020936
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001327, 0.0001316
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0050794, 0.0051087
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054787, 0.0054893
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025896, 0.0025860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003429, upper bound: 0.0003448
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003490, upper bound: 0.0003393
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027500, 0.0027406
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004867, 0.0004858
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034554, 0.0034443
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008308, 0.0008319
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005264, 0.0005280
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020991, 0.0020920
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001324, 0.0001318
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0050792, 0.0051039
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054698, 0.0054860
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025887, 0.0025816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003393, upper bound: 0.0003490
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003448, upper bound: 0.0003429
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027513, 0.0027424
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004868, 0.0004865
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034573, 0.0034470
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008324, 0.0008315
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005268, 0.0005282
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0021002, 0.0020934
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001325, 0.0001317
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0050804, 0.0051049
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054751, 0.0054879
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025894, 0.0025844

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003393, upper bound: 0.0003490
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003448, upper bound: 0.0003429
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027529, 0.0027386
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004873, 0.0004855
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034591, 0.0034420
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008300, 0.0008324
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005260, 0.0005285
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0021013, 0.0020905
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001326, 0.0001318
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0050776, 0.0051063
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054656, 0.0054916
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025916, 0.0025794

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003410, upper bound: 0.0003472
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003469, upper bound: 0.0003413
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027542, 0.0027405
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004875, 0.0004862
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034611, 0.0034446
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008316, 0.0008323
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005265, 0.0005288
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0021024, 0.0020919
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001327, 0.0001316
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0050788, 0.0051074
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054709, 0.0054946
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025930, 0.0025821

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003410, upper bound: 0.0003472
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003469, upper bound: 0.0003413
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027369, 0.0027541
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004853, 0.0004876
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034391, 0.0034611
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008330, 0.0008301
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005288, 0.0005256
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020891, 0.0021023
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001317, 0.0001327
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0051065, 0.0050766
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054947, 0.0054618
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025777, 0.0025930

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003413, upper bound: 0.0003469
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003472, upper bound: 0.0003410
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027387, 0.0027560
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004854, 0.0004883
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034420, 0.0034637
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008346, 0.0008291
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005293, 0.0005260
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020906, 0.0021037
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001318, 0.0001325
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0051078, 0.0050780
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0055000, 0.0054656
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025793, 0.0025957

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003413, upper bound: 0.0003469
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003472, upper bound: 0.0003410
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027393, 0.0027514
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004857, 0.0004869
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034428, 0.0034575
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008324, 0.0008308
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005282, 0.0005261
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020910, 0.0021003
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001318, 0.0001326
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0051042, 0.0050780
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054880, 0.0054679
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025808, 0.0025895

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003429, upper bound: 0.0003448
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003490, upper bound: 0.0003393
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027407, 0.0027533
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004857, 0.0004876
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034444, 0.0034601
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008340, 0.0008300
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005287, 0.0005264
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020921, 0.0021017
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001319, 0.0001324
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0051054, 0.0050794
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054933, 0.0054697
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025816, 0.0025922

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003429, upper bound: 0.0003448
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003490, upper bound: 0.0003393
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027390, 0.0027525
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004860, 0.0004866
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034432, 0.0034587
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008308, 0.0008321
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005284, 0.0005263
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020909, 0.0021011
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001317, 0.0001327
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0051078, 0.0050770
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054894, 0.0054692
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025811, 0.0025897

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003393, upper bound: 0.0003490
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003448, upper bound: 0.0003429
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027407, 0.0027544
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004863, 0.0004873
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034457, 0.0034613
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008324, 0.0008316
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005289, 0.0005267
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020922, 0.0021025
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001318, 0.0001326
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0051091, 0.0050785
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054947, 0.0054733
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025832, 0.0025924

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003393, upper bound: 0.0003490
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003448, upper bound: 0.0003429
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027416, 0.0027498
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004864, 0.0004862
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034461, 0.0034549
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008299, 0.0008323
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005277, 0.0005268
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020928, 0.0020990
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001318, 0.0001326
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0051058, 0.0050787
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054822, 0.0054755
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025848, 0.0025863

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003410, upper bound: 0.0003472
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003469, upper bound: 0.0003413
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027428, 0.0027517
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004867, 0.0004869
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034484, 0.0034575
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008315, 0.0008322
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005282, 0.0005272
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020938, 0.0021004
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001319, 0.0001324
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0051071, 0.0050803
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054876, 0.0054786
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025861, 0.0025890

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003410, upper bound: 0.0003472
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003469, upper bound: 0.0003413
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027494, 0.0027428
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004862, 0.0004867
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034543, 0.0034484
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008322, 0.0008308
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005272, 0.0005277
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020986, 0.0020938
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001324, 0.0001319
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0050803, 0.0051042
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054786, 0.0054816
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025863, 0.0025861

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003413, upper bound: 0.0003469
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003472, upper bound: 0.0003410
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027498, 0.0027447
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004862, 0.0004874
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034549, 0.0034510
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008338, 0.0008299
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005276, 0.0005277
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020990, 0.0020952
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001325, 0.0001318
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0050815, 0.0051058
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054840, 0.0054822
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025863, 0.0025888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003413, upper bound: 0.0003469
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003472, upper bound: 0.0003410
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027517, 0.0027407
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004866, 0.0004863
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034574, 0.0034457
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008316, 0.0008318
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005267, 0.0005281
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0021004, 0.0020922
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001326, 0.0001318
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0050785, 0.0051061
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054733, 0.0054863
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025885, 0.0025832

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003429, upper bound: 0.0003448
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003490, upper bound: 0.0003393
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027525, 0.0027425
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004866, 0.0004870
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034587, 0.0034484
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008332, 0.0008308
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005271, 0.0005284
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0021011, 0.0020936
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001327, 0.0001317
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0050798, 0.0051078
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054787, 0.0054894
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025897, 0.0025859

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003429, upper bound: 0.0003448
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003490, upper bound: 0.0003393
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027502, 0.0027407
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004869, 0.0004857
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034557, 0.0034444
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008300, 0.0008327
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005264, 0.0005280
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020992, 0.0020921
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001324, 0.0001319
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0050794, 0.0051033
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054697, 0.0054860
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025887, 0.0025816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003393, upper bound: 0.0003490
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003448, upper bound: 0.0003429
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027514, 0.0027425
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004869, 0.0004864
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034575, 0.0034470
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008316, 0.0008324
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005268, 0.0005282
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0021003, 0.0020935
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001325, 0.0001318
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0050806, 0.0051042
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054751, 0.0054880
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025895, 0.0025843

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003393, upper bound: 0.0003490
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003448, upper bound: 0.0003429
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027529, 0.0027387
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004874, 0.0004854
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034592, 0.0034420
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008291, 0.0008333
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005260, 0.0005285
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0021013, 0.0020906
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001325, 0.0001318
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0050780, 0.0051055
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054656, 0.0054916
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025917, 0.0025793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003410, upper bound: 0.0003472
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003469, upper bound: 0.0003412
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027541, 0.0027406
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004876, 0.0004861
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034611, 0.0034446
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008307, 0.0008330
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005265, 0.0005288
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0021023, 0.0020920
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001327, 0.0001317
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0050792, 0.0051065
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054709, 0.0054947
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025930, 0.0025820

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003410, upper bound: 0.0003472
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003469, upper bound: 0.0003412
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027368, 0.0027542
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004854, 0.0004875
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034391, 0.0034611
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008323, 0.0008309
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005288, 0.0005256
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020890, 0.0021024
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001316, 0.0001327
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0051074, 0.0050759
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054946, 0.0054618
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025777, 0.0025930

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003413, upper bound: 0.0003469
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003472, upper bound: 0.0003410
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027386, 0.0027561
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004855, 0.0004882
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034420, 0.0034637
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008339, 0.0008300
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005293, 0.0005260
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020905, 0.0021038
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001317, 0.0001326
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0051087, 0.0050776
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0055000, 0.0054656
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025794, 0.0025957

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003413, upper bound: 0.0003469
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003472, upper bound: 0.0003410
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027392, 0.0027513
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004858, 0.0004868
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034427, 0.0034573
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008315, 0.0008318
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005282, 0.0005261
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020909, 0.0021002
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001317, 0.0001326
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0051049, 0.0050774
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054879, 0.0054679
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025808, 0.0025894

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003429, upper bound: 0.0003448
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003490, upper bound: 0.0003393
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027406, 0.0027532
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004858, 0.0004875
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034443, 0.0034599
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008331, 0.0008308
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005287, 0.0005264
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020920, 0.0021016
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001318, 0.0001324
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0051062, 0.0050792
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054933, 0.0054698
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025816, 0.0025921

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003429, upper bound: 0.0003448
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003490, upper bound: 0.0003393
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027390, 0.0027526
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004861, 0.0004865
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034433, 0.0034587
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008301, 0.0008329
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005284, 0.0005263
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020909, 0.0021011
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001316, 0.0001328
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0051087, 0.0050767
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054893, 0.0054692
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025811, 0.0025896

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003393, upper bound: 0.0003490
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003448, upper bound: 0.0003429
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027406, 0.0027544
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004864, 0.0004872
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034458, 0.0034613
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008317, 0.0008325
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005288, 0.0005267
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020922, 0.0021025
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001317, 0.0001326
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0051099, 0.0050781
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054947, 0.0054733
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025833, 0.0025923

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003393, upper bound: 0.0003490
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003448, upper bound: 0.0003429
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027415, 0.0027499
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004865, 0.0004861
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034461, 0.0034549
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008290, 0.0008333
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005277, 0.0005268
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020928, 0.0020990
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001317, 0.0001326
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0051062, 0.0050783
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054822, 0.0054755
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025849, 0.0025862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003410, upper bound: 0.0003472
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003469, upper bound: 0.0003413
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0046435, -0.0012692, -0.0046435, -0.0012692, -0.0027428, 0.0027517
1: -0.0047322, -0.0041408, -0.0047322, -0.0041408, -0.0004868, 0.0004868
2: 0.0086693, 0.0129236, 0.0086693, 0.0129236, -0.0034484, 0.0034575
3: 1.0083368, 1.0093260, 1.0083368, 1.0093260, -0.0008306, 0.0008331
4: -0.0037052, -0.0030525, -0.0037052, -0.0030525, -0.0005282, 0.0005272
5: 0.0003986, 0.0029757, 0.0003986, 0.0029757, -0.0020937, 0.0021004
6: -0.0025740, -0.0024289, -0.0025740, -0.0024289, -0.0001318, 0.0001324
7: -0.0116998, -0.0055757, -0.0116998, -0.0055757, -0.0051075, 0.0050799
8: -0.0075350, -0.0007426, -0.0075350, -0.0007426, -0.0054875, 0.0054786
9: -0.0037703, -0.0005713, -0.0037703, -0.0005713, -0.0025862, 0.0025890

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003410, upper bound: 0.0003472
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003469, upper bound: 0.0003413
time: 0.88 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.31 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003413, upper bound: 0.0003469
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003472, upper bound: 0.0003410
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003413, upper bound: 0.0003469
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003472, upper bound: 0.0003410
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003429, upper bound: 0.0003448
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003490, upper bound: 0.0003393
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003429, upper bound: 0.0003448
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003490, upper bound: 0.0003393
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003393, upper bound: 0.0003490
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003448, upper bound: 0.0003429
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003393, upper bound: 0.0003490
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003448, upper bound: 0.0003429
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003410, upper bound: 0.0003472
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003469, upper bound: 0.0003413
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003410, upper bound: 0.0003472
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003469, upper bound: 0.0003413
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003413, upper bound: 0.0003469
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003472, upper bound: 0.0003410
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003413, upper bound: 0.0003469
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003472, upper bound: 0.0003410
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003429, upper bound: 0.0003448
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003490, upper bound: 0.0003393
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003429, upper bound: 0.0003448
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003490, upper bound: 0.0003393
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003393, upper bound: 0.0003490
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003448, upper bound: 0.0003429
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003393, upper bound: 0.0003490
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003448, upper bound: 0.0003429
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003410, upper bound: 0.0003472
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003469, upper bound: 0.0003413
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003410, upper bound: 0.0003472
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003469, upper bound: 0.0003413
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003413, upper bound: 0.0003469
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003472, upper bound: 0.0003410
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003413, upper bound: 0.0003469
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003472, upper bound: 0.0003410
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003429, upper bound: 0.0003448
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003490, upper bound: 0.0003393
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003429, upper bound: 0.0003448
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003490, upper bound: 0.0003393
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003393, upper bound: 0.0003490
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003448, upper bound: 0.0003429
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003393, upper bound: 0.0003490
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003448, upper bound: 0.0003429
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003410, upper bound: 0.0003472
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003469, upper bound: 0.0003412
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003410, upper bound: 0.0003472
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003469, upper bound: 0.0003412
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003413, upper bound: 0.0003469
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003472, upper bound: 0.0003410
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003413, upper bound: 0.0003469
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003472, upper bound: 0.0003410
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003429, upper bound: 0.0003448
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003490, upper bound: 0.0003393
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003429, upper bound: 0.0003448
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003490, upper bound: 0.0003393
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003393, upper bound: 0.0003490
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003448, upper bound: 0.0003429
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003393, upper bound: 0.0003490
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003448, upper bound: 0.0003429
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003410, upper bound: 0.0003472
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003469, upper bound: 0.0003413
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003410, upper bound: 0.0003472
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -0.0003469, upper bound: 0.0003413

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.10 + 205.62 = 208.72 seconds
