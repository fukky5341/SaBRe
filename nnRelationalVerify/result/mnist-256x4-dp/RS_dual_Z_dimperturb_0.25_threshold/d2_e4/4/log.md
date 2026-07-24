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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0029717, 0.0052943, 0.0029717, 0.0052943, -0.0019721, 0.0019721)
1: (0.0016688, 0.0020735, 0.0016688, 0.0020735, -0.0003521, 0.0003521)
2: (0.0114147, 0.0127169, 0.0114147, 0.0127169, -0.0011186, 0.0011186)
3: (-0.0028020, -0.0014488, -0.0028020, -0.0014488, -0.0011167, 0.0011167)
4: (-0.0023906, -0.0010036, -0.0023906, -0.0010036, -0.0011296, 0.0011296)
5: (0.0050241, 0.0063682, 0.0050241, 0.0063682, -0.0011288, 0.0011288)
6: (-0.0022545, 0.0029669, -0.0022545, 0.0029669, -0.0042863, 0.0042863)
7: (-0.0066880, 0.0004550, -0.0066880, 0.0004550, -0.0058277, 0.0058277)
8: (0.9845665, 0.9895484, 0.9845665, 0.9895484, -0.0040649, 0.0040649)
9: (-0.0063873, -0.0018484, -0.0063873, -0.0018484, -0.0036976, 0.0036976)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.53 + 1.65 = 3.18 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0019997, upper bound: 0.0019997

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019347, upper bound: 0.0019637
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019637, upper bound: 0.0019347
time: 0.59 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.38 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.38
Output dim: 8, lower bound: -0.0019347, upper bound: 0.0019637
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.38
Output dim: 8, lower bound: -0.0019637, upper bound: 0.0019347

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0029717, 0.0052943, 0.0029717, 0.0052943, -0.0019632, 0.0019668
1: 0.0016688, 0.0020735, 0.0016688, 0.0020735, -0.0003508, 0.0003513
2: 0.0114147, 0.0127169, 0.0114147, 0.0127169, -0.0011157, 0.0011137
3: -0.0028020, -0.0014488, -0.0028020, -0.0014488, -0.0011136, 0.0011115
4: -0.0023906, -0.0010036, -0.0023906, -0.0010036, -0.0011240, 0.0011263
5: 0.0050241, 0.0063682, 0.0050241, 0.0063682, -0.0011257, 0.0011236
6: -0.0022545, 0.0029669, -0.0022545, 0.0029669, -0.0042738, 0.0042655
7: -0.0066880, 0.0004550, -0.0066880, 0.0004550, -0.0057991, 0.0058106
8: 0.9845665, 0.9895484, 0.9845665, 0.9895484, -0.0040449, 0.0040529
9: -0.0063873, -0.0018484, -0.0063873, -0.0018484, -0.0036866, 0.0036793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016798, upper bound: 0.0016798
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016798, upper bound: 0.0016798
time: 0.57 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0029717, 0.0052943, 0.0029717, 0.0052943, -0.0019668, 0.0019632
1: 0.0016688, 0.0020735, 0.0016688, 0.0020735, -0.0003513, 0.0003508
2: 0.0114147, 0.0127169, 0.0114147, 0.0127169, -0.0011137, 0.0011157
3: -0.0028020, -0.0014488, -0.0028020, -0.0014488, -0.0011115, 0.0011136
4: -0.0023906, -0.0010036, -0.0023906, -0.0010036, -0.0011263, 0.0011240
5: 0.0050241, 0.0063682, 0.0050241, 0.0063682, -0.0011236, 0.0011257
6: -0.0022545, 0.0029669, -0.0022545, 0.0029669, -0.0042655, 0.0042738
7: -0.0066880, 0.0004550, -0.0066880, 0.0004550, -0.0058106, 0.0057991
8: 0.9845665, 0.9895484, 0.9845665, 0.9895484, -0.0040529, 0.0040449
9: -0.0063873, -0.0018484, -0.0063873, -0.0018484, -0.0036793, 0.0036866

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016798, upper bound: 0.0016798
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016798, upper bound: 0.0016798
time: 0.58 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.76 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.76
Output dim: 8, lower bound: -0.0016798, upper bound: 0.0016798
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.76
Output dim: 8, lower bound: -0.0016798, upper bound: 0.0016798
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.76
Output dim: 8, lower bound: -0.0016798, upper bound: 0.0016798
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.76
Output dim: 8, lower bound: -0.0016798, upper bound: 0.0016798

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0029717, 0.0052943, 0.0029717, 0.0052943, -0.0019593, 0.0019601
1: 0.0016688, 0.0020735, 0.0016688, 0.0020735, -0.0003503, 0.0003505
2: 0.0114147, 0.0127169, 0.0114147, 0.0127169, -0.0011119, 0.0011115
3: -0.0028020, -0.0014488, -0.0028020, -0.0014488, -0.0011099, 0.0011095
4: -0.0023906, -0.0010036, -0.0023906, -0.0010036, -0.0011217, 0.0011222
5: 0.0050241, 0.0063682, 0.0050241, 0.0063682, -0.0011217, 0.0011213
6: -0.0022545, 0.0029669, -0.0022545, 0.0029669, -0.0042582, 0.0042564
7: -0.0066880, 0.0004550, -0.0066880, 0.0004550, -0.0057871, 0.0057895
8: 0.9845665, 0.9895484, 0.9845665, 0.9895484, -0.0040362, 0.0040379
9: -0.0063873, -0.0018484, -0.0063873, -0.0018484, -0.0036731, 0.0036715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015632, upper bound: 0.0015673
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015673, upper bound: 0.0015632
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0029717, 0.0052943, 0.0029717, 0.0052943, -0.0019632, 0.0019629
1: 0.0016688, 0.0020735, 0.0016688, 0.0020735, -0.0003508, 0.0003509
2: 0.0114147, 0.0127169, 0.0114147, 0.0127169, -0.0011135, 0.0011137
3: -0.0028020, -0.0014488, -0.0028020, -0.0014488, -0.0011116, 0.0011115
4: -0.0023906, -0.0010036, -0.0023906, -0.0010036, -0.0011240, 0.0011239
5: 0.0050241, 0.0063682, 0.0050241, 0.0063682, -0.0011234, 0.0011236
6: -0.0022545, 0.0029669, -0.0022545, 0.0029669, -0.0042648, 0.0042655
7: -0.0066880, 0.0004550, -0.0066880, 0.0004550, -0.0057991, 0.0057985
8: 0.9845665, 0.9895484, 0.9845665, 0.9895484, -0.0040449, 0.0040442
9: -0.0063873, -0.0018484, -0.0063873, -0.0018484, -0.0036788, 0.0036793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015632, upper bound: 0.0015673
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015673, upper bound: 0.0015632
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0029717, 0.0052943, 0.0029717, 0.0052943, -0.0019629, 0.0019569
1: 0.0016688, 0.0020735, 0.0016688, 0.0020735, -0.0003509, 0.0003500
2: 0.0114147, 0.0127169, 0.0114147, 0.0127169, -0.0011102, 0.0011135
3: -0.0028020, -0.0014488, -0.0028020, -0.0014488, -0.0011081, 0.0011116
4: -0.0023906, -0.0010036, -0.0023906, -0.0010036, -0.0011239, 0.0011202
5: 0.0050241, 0.0063682, 0.0050241, 0.0063682, -0.0011199, 0.0011234
6: -0.0022545, 0.0029669, -0.0022545, 0.0029669, -0.0042508, 0.0042648
7: -0.0066880, 0.0004550, -0.0066880, 0.0004550, -0.0057985, 0.0057794
8: 0.9845665, 0.9895484, 0.9845665, 0.9895484, -0.0040442, 0.0040308
9: -0.0063873, -0.0018484, -0.0063873, -0.0018484, -0.0036666, 0.0036788

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015632, upper bound: 0.0015673
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015673, upper bound: 0.0015632
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0029717, 0.0052943, 0.0029717, 0.0052943, -0.0019668, 0.0019593
1: 0.0016688, 0.0020735, 0.0016688, 0.0020735, -0.0003513, 0.0003503
2: 0.0114147, 0.0127169, 0.0114147, 0.0127169, -0.0011115, 0.0011157
3: -0.0028020, -0.0014488, -0.0028020, -0.0014488, -0.0011095, 0.0011136
4: -0.0023906, -0.0010036, -0.0023906, -0.0010036, -0.0011263, 0.0011217
5: 0.0050241, 0.0063682, 0.0050241, 0.0063682, -0.0011213, 0.0011257
6: -0.0022545, 0.0029669, -0.0022545, 0.0029669, -0.0042564, 0.0042738
7: -0.0066880, 0.0004550, -0.0066880, 0.0004550, -0.0058106, 0.0057871
8: 0.9845665, 0.9895484, 0.9845665, 0.9895484, -0.0040529, 0.0040362
9: -0.0063873, -0.0018484, -0.0063873, -0.0018484, -0.0036715, 0.0036866

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015632, upper bound: 0.0015673
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015673, upper bound: 0.0015632
time: 0.69 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.99 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.99
Output dim: 8, lower bound: -0.0015632, upper bound: 0.0015673
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.99
Output dim: 8, lower bound: -0.0015673, upper bound: 0.0015632
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.99
Output dim: 8, lower bound: -0.0015632, upper bound: 0.0015673
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.99
Output dim: 8, lower bound: -0.0015673, upper bound: 0.0015632
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.99
Output dim: 8, lower bound: -0.0015632, upper bound: 0.0015673
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.99
Output dim: 8, lower bound: -0.0015673, upper bound: 0.0015632
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.99
Output dim: 8, lower bound: -0.0015632, upper bound: 0.0015673
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.99
Output dim: 8, lower bound: -0.0015673, upper bound: 0.0015632

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.18 + 18.45 = 21.63 seconds
