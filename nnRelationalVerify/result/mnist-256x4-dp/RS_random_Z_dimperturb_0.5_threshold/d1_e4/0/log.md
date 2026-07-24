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
0: (-0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025852, 0.0025852)
1: (-0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004653, 0.0004653)
2: (0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0032465, 0.0032465)
3: (1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0008559, 0.0008559)
4: (-0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004962, 0.0004962)
5: (0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019730, 0.0019730)
6: (-0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001328, 0.0001328)
7: (-0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0048422, 0.0048422)
8: (-0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0051588, 0.0051588)
9: (-0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0024405, 0.0024405)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.41 + 1.81 = 3.22 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0004453, upper bound: 0.0004454

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004362, upper bound: 0.0004210
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004210, upper bound: 0.0004362
time: 1.04 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.02 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.02
Output dim: 3, lower bound: -0.0004362, upper bound: 0.0004210
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.02
Output dim: 3, lower bound: -0.0004210, upper bound: 0.0004362

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025849, 0.0025852
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004590, 0.0004603
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0032447, 0.0032452
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0008403, 0.0008346
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004954, 0.0004951
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019728, 0.0019730
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001306, 0.0001302
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0048343, 0.0048317
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0051457, 0.0051412
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0024278, 0.0024307

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003920, upper bound: 0.0003792
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003910, upper bound: 0.0003804
time: 1.10 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025852, 0.0025849
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004603, 0.0004590
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0032452, 0.0032447
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0008346, 0.0008403
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004951, 0.0004954
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019730, 0.0019728
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001302, 0.0001306
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0048317, 0.0048343
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0051412, 0.0051457
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0024307, 0.0024278

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003804, upper bound: 0.0003910
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003792, upper bound: 0.0003920
time: 0.95 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.15 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 3, lower bound: -0.0003920, upper bound: 0.0003792
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 3, lower bound: -0.0003910, upper bound: 0.0003804
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 3, lower bound: -0.0003804, upper bound: 0.0003910
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 3, lower bound: -0.0003792, upper bound: 0.0003920

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025834, 0.0025839
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004577, 0.0004602
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0032442, 0.0032448
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0008432, 0.0008292
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004953, 0.0004949
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019717, 0.0019721
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001310, 0.0001294
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0048241, 0.0048281
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0051438, 0.0051380
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0024253, 0.0024293

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003462, upper bound: 0.0003384
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003463, upper bound: 0.0003384
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025836, 0.0025852
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004590, 0.0004591
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0032443, 0.0032452
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0008348, 0.0008346
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004952, 0.0004951
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019720, 0.0019730
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001299, 0.0001302
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0048343, 0.0048214
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0051424, 0.0051412
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0024278, 0.0024282

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003885, upper bound: 0.0003776
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003885, upper bound: 0.0003769
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025836, 0.0025836
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004591, 0.0004590
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0032447, 0.0032443
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0008384, 0.0008348
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004950, 0.0004952
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019719, 0.0019720
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001305, 0.0001299
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0048214, 0.0048294
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0051395, 0.0051424
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0024282, 0.0024264

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003769, upper bound: 0.0003885
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003776, upper bound: 0.0003885
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025839, 0.0025849
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004603, 0.0004577
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0032448, 0.0032447
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0008292, 0.0008403
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004949, 0.0004954
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019721, 0.0019728
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001294, 0.0001306
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0048317, 0.0048241
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0051380, 0.0051457
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0024307, 0.0024253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003752, upper bound: 0.0003845
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003735, upper bound: 0.0003871
time: 1.00 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 5.10 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 5.10
Output dim: 3, lower bound: -0.0003462, upper bound: 0.0003384
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 5.10
Output dim: 3, lower bound: -0.0003463, upper bound: 0.0003384
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.10
Output dim: 3, lower bound: -0.0003885, upper bound: 0.0003776
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.10
Output dim: 3, lower bound: -0.0003885, upper bound: 0.0003769
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.10
Output dim: 3, lower bound: -0.0003769, upper bound: 0.0003885
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.10
Output dim: 3, lower bound: -0.0003776, upper bound: 0.0003885
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.10
Output dim: 3, lower bound: -0.0003752, upper bound: 0.0003845
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.10
Output dim: 3, lower bound: -0.0003735, upper bound: 0.0003871

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025647, 0.0025651
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004561, 0.0004561
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0032191, 0.0032192
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0008313, 0.0008313
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004913, 0.0004913
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019574, 0.0019576
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001298, 0.0001300
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0048093, 0.0047971
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0051023, 0.0051021
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0024097, 0.0024095

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003768, upper bound: 0.0003704
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003811, upper bound: 0.0003634
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025636, 0.0025662
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004560, 0.0004562
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0032182, 0.0032200
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0008315, 0.0008311
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004914, 0.0004912
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019566, 0.0019584
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001298, 0.0001301
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0048103, 0.0047964
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0051033, 0.0051014
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0024092, 0.0024100

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003706, upper bound: 0.0003672
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003788, upper bound: 0.0003593
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025646, 0.0025636
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004562, 0.0004561
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0032195, 0.0032182
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0008351, 0.0008315
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004911, 0.0004914
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019573, 0.0019566
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001304, 0.0001298
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0047964, 0.0048051
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0050999, 0.0051033
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0024100, 0.0024082

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003609, upper bound: 0.0003706
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003596, upper bound: 0.0003718
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025636, 0.0025647
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004561, 0.0004561
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0032186, 0.0032191
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0008350, 0.0008313
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004912, 0.0004913
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019566, 0.0019574
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001304, 0.0001298
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0047971, 0.0048048
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0051004, 0.0051023
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0024095, 0.0024083

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003596, upper bound: 0.0003788
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003681, upper bound: 0.0003706
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025792, 0.0025805
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004592, 0.0004568
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0032407, 0.0032409
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0008242, 0.0008343
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004947, 0.0004951
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019687, 0.0019697
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001285, 0.0001297
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0048137, 0.0048053
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0051374, 0.0051450
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0024298, 0.0024245

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003558, upper bound: 0.0003713
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003613, upper bound: 0.0003644
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025795, 0.0025803
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004594, 0.0004567
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0032410, 0.0032406
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0008232, 0.0008354
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004946, 0.0004951
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019689, 0.0019695
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001284, 0.0001297
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0048133, 0.0048060
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0051373, 0.0051451
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0024299, 0.0024244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003734, upper bound: 0.0003868
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003734, upper bound: 0.0003871
time: 1.02 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.24 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 3, lower bound: -0.0003768, upper bound: 0.0003704
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 3, lower bound: -0.0003811, upper bound: 0.0003634
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 3, lower bound: -0.0003706, upper bound: 0.0003672
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 3, lower bound: -0.0003788, upper bound: 0.0003593
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 3, lower bound: -0.0003609, upper bound: 0.0003706
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 3, lower bound: -0.0003596, upper bound: 0.0003718
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 3, lower bound: -0.0003596, upper bound: 0.0003788
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 3, lower bound: -0.0003681, upper bound: 0.0003706
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 3, lower bound: -0.0003558, upper bound: 0.0003713
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 3, lower bound: -0.0003613, upper bound: 0.0003644
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 3, lower bound: -0.0003734, upper bound: 0.0003868
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 3, lower bound: -0.0003734, upper bound: 0.0003871

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025889, 0.0025876
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004568, 0.0004556
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0032522, 0.0032489
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007909, 0.0007921
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004957, 0.0004963
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019763, 0.0019751
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001264, 0.0001267
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0048288, 0.0048169
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0051440, 0.0051511
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0024308, 0.0024268

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003605, upper bound: 0.0003608
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003667, upper bound: 0.0003523
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025871, 0.0025894
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004557, 0.0004566
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0032489, 0.0032521
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007911, 0.0007904
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004962, 0.0004956
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019749, 0.0019766
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001265, 0.0001267
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0048290, 0.0048164
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0051504, 0.0051439
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0024269, 0.0024304

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
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003360, upper bound: 0.0003233
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003361, upper bound: 0.0003231
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025038, 0.0025014
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004344, 0.0004330
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031326, 0.0031284
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007938, 0.0007973
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004753, 0.0004760
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019102, 0.0019081
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001293, 0.0001296
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0047415, 0.0047346
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0049208, 0.0049287
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023200, 0.0023157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003666, upper bound: 0.0003614
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003647, upper bound: 0.0003629
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0024988, 0.0025063
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004329, 0.0004346
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031266, 0.0031348
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007968, 0.0007935
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004762, 0.0004751
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019064, 0.0019120
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001292, 0.0001296
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0047470, 0.0047281
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0049310, 0.0049193
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023151, 0.0023208

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003782, upper bound: 0.0003582
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003781, upper bound: 0.0003582
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025254, 0.0025155
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004511, 0.0004508
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031700, 0.0031602
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0008312, 0.0008274
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004827, 0.0004838
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019276, 0.0019202
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001282, 0.0001266
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0046895, 0.0047221
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0050137, 0.0050233
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023726, 0.0023688

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003606, upper bound: 0.0003678
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003597, upper bound: 0.0003703
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025165, 0.0025238
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004509, 0.0004509
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031615, 0.0031683
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0008310, 0.0008276
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004834, 0.0004829
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019209, 0.0019264
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001272, 0.0001277
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0047159, 0.0046982
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0050193, 0.0050171
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023706, 0.0023707

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003166, upper bound: 0.0003232
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003166, upper bound: 0.0003232
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025041, 0.0024998
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004343, 0.0004329
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031330, 0.0031275
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007973, 0.0007966
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004751, 0.0004761
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019104, 0.0019071
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001299, 0.0001292
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0047288, 0.0047437
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0049178, 0.0049291
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023198, 0.0023139

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003178, upper bound: 0.0003337
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003182, upper bound: 0.0003336
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0024988, 0.0025049
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004328, 0.0004343
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031270, 0.0031340
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007988, 0.0007936
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004760, 0.0004752
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019063, 0.0019110
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001298, 0.0001293
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0047359, 0.0047365
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0049268, 0.0049198
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023152, 0.0023189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003636, upper bound: 0.0003632
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003616, upper bound: 0.0003660
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0022845, 0.0022430
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0003929, 0.0003860
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0028474, 0.0027962
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007785, 0.0007886
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004229, 0.0004306
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0017419, 0.0017104
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001279, 0.0001273
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0043368, 0.0044169
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0043695, 0.0044488
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0020887, 0.0020518

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003526, upper bound: 0.0003687
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003531, upper bound: 0.0003687
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0022414, 0.0022862
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0003883, 0.0003906
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0027960, 0.0028477
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007792, 0.0007884
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004302, 0.0004233
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0017092, 0.0017431
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001261, 0.0001292
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0044292, 0.0043263
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0044423, 0.0043767
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0020569, 0.0020838

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003485, upper bound: 0.0003571
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003540, upper bound: 0.0003506
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025915, 0.0025934
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004607, 0.0004578
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0032590, 0.0032598
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0008165, 0.0008281
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004978, 0.0004981
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019784, 0.0019798
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001285, 0.0001298
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0048285, 0.0048189
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0051675, 0.0051740
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0024418, 0.0024366

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003703, upper bound: 0.0003842
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003707, upper bound: 0.0003842
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025925, 0.0025925
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004606, 0.0004580
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0032602, 0.0032588
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0008165, 0.0008286
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004977, 0.0004982
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019791, 0.0019791
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001285, 0.0001298
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0048277, 0.0048216
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0051666, 0.0051754
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0024421, 0.0024362

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003690, upper bound: 0.0003782
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003676, upper bound: 0.0003824
time: 1.03 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.28 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 3, lower bound: -0.0003605, upper bound: 0.0003608
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 3, lower bound: -0.0003667, upper bound: 0.0003523
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.28
Output dim: 3, lower bound: -0.0003360, upper bound: 0.0003233
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.28
Output dim: 3, lower bound: -0.0003361, upper bound: 0.0003231
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 3, lower bound: -0.0003666, upper bound: 0.0003614
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 3, lower bound: -0.0003647, upper bound: 0.0003629
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 3, lower bound: -0.0003782, upper bound: 0.0003582
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 3, lower bound: -0.0003781, upper bound: 0.0003582
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 3, lower bound: -0.0003606, upper bound: 0.0003678
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 3, lower bound: -0.0003597, upper bound: 0.0003703
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.28
Output dim: 3, lower bound: -0.0003166, upper bound: 0.0003232
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.28
Output dim: 3, lower bound: -0.0003166, upper bound: 0.0003232
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.28
Output dim: 3, lower bound: -0.0003178, upper bound: 0.0003337
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.28
Output dim: 3, lower bound: -0.0003182, upper bound: 0.0003336
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 3, lower bound: -0.0003636, upper bound: 0.0003632
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 3, lower bound: -0.0003616, upper bound: 0.0003660
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 3, lower bound: -0.0003526, upper bound: 0.0003687
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 3, lower bound: -0.0003531, upper bound: 0.0003687
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 3, lower bound: -0.0003485, upper bound: 0.0003571
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.28
Output dim: 3, lower bound: -0.0003540, upper bound: 0.0003506
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 3, lower bound: -0.0003703, upper bound: 0.0003842
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 3, lower bound: -0.0003707, upper bound: 0.0003842
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 3, lower bound: -0.0003690, upper bound: 0.0003782
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 3, lower bound: -0.0003676, upper bound: 0.0003824

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025301, 0.0025226
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004368, 0.0004335
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031702, 0.0031593
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007534, 0.0007591
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004804, 0.0004822
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019308, 0.0019248
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001260, 0.0001263
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0047599, 0.0047563
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0049752, 0.0049953
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023516, 0.0023405

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003401, upper bound: 0.0003469
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003467, upper bound: 0.0003407
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025239, 0.0025276
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004348, 0.0004350
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031626, 0.0031651
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007564, 0.0007548
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004813, 0.0004810
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019261, 0.0019287
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001258, 0.0001263
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0047648, 0.0047486
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0049845, 0.0049826
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023447, 0.0023451

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003657, upper bound: 0.0003489
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003630, upper bound: 0.0003512
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0024991, 0.0024970
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004333, 0.0004321
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031285, 0.0031245
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007889, 0.0007913
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004750, 0.0004757
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019068, 0.0019049
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001284, 0.0001286
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0047234, 0.0047166
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0049201, 0.0049280
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023190, 0.0023149

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003481, upper bound: 0.0003419
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003473, upper bound: 0.0003429
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0024994, 0.0024967
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004335, 0.0004319
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031287, 0.0031243
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007878, 0.0007923
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004750, 0.0004758
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019070, 0.0019047
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001283, 0.0001286
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0047228, 0.0047165
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0049201, 0.0049281
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023191, 0.0023148

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003637, upper bound: 0.0003612
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003595, upper bound: 0.0003619
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025096, 0.0025178
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004357, 0.0004377
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031439, 0.0031535
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007915, 0.0007882
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004797, 0.0004783
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019150, 0.0019211
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001292, 0.0001296
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0047692, 0.0047488
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0049691, 0.0049541
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023323, 0.0023394

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003617, upper bound: 0.0003417
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003604, upper bound: 0.0003427
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025102, 0.0025170
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004359, 0.0004379
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031453, 0.0031530
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007910, 0.0007882
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004798, 0.0004786
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019154, 0.0019204
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001292, 0.0001297
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0047676, 0.0047505
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0049701, 0.0049572
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023337, 0.0023405

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003771, upper bound: 0.0003552
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003752, upper bound: 0.0003573
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0024117, 0.0024137
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004396, 0.0004421
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0030360, 0.0030422
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0008337, 0.0008288
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004659, 0.0004643
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0018415, 0.0018433
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001238, 0.0001222
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0044657, 0.0044862
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0048492, 0.0048294
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0022860, 0.0022965

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003160, upper bound: 0.0003215
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003160, upper bound: 0.0003214
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0024236, 0.0024070
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004424, 0.0004405
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0030520, 0.0030312
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0008333, 0.0008299
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004640, 0.0004670
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0018506, 0.0018379
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001238, 0.0001227
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0044661, 0.0044983
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0048282, 0.0048588
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023003, 0.0022867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003596, upper bound: 0.0003702
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003596, upper bound: 0.0003702
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0024808, 0.0024898
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004299, 0.0004325
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031049, 0.0031168
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007958, 0.0007882
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004736, 0.0004720
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0018926, 0.0018996
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001297, 0.0001293
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0047153, 0.0047125
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0049039, 0.0048874
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023002, 0.0023085

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003633, upper bound: 0.0003614
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003624, upper bound: 0.0003630
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0024837, 0.0024877
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004310, 0.0004315
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031099, 0.0031135
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007930, 0.0007906
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004731, 0.0004729
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0018949, 0.0018979
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001297, 0.0001293
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0047135, 0.0047159
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0048976, 0.0048969
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023048, 0.0023048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003412, upper bound: 0.0003514
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003478, upper bound: 0.0003463
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0022683, 0.0022257
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0003894, 0.0003824
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0028246, 0.0027726
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007754, 0.0007857
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004190, 0.0004268
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0017293, 0.0016970
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001278, 0.0001272
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0043115, 0.0043923
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0043293, 0.0044095
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0020701, 0.0020327

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003336, upper bound: 0.0003472
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003327, upper bound: 0.0003485
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0022683, 0.0022268
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0003892, 0.0003825
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0028246, 0.0027734
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007756, 0.0007854
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004191, 0.0004268
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0017293, 0.0016978
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001278, 0.0001272
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0043121, 0.0043919
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0043300, 0.0044092
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0020703, 0.0020332

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003378, upper bound: 0.0003612
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003458, upper bound: 0.0003566
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0022727, 0.0023158
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0003899, 0.0003912
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0028403, 0.0028887
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007358, 0.0007452
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004369, 0.0004305
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0017336, 0.0017661
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001228, 0.0001259
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0044502, 0.0043474
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0045100, 0.0044509
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0020891, 0.0021125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003308, upper bound: 0.0003472
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003389, upper bound: 0.0003422
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025722, 0.0025731
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004577, 0.0004547
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0032326, 0.0032324
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0008133, 0.0008251
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004935, 0.0004939
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019636, 0.0019641
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001284, 0.0001296
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0048023, 0.0047935
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0051218, 0.0051290
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0024208, 0.0024151

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003524, upper bound: 0.0003627
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003512, upper bound: 0.0003635
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025716, 0.0025742
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004572, 0.0004549
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0032314, 0.0032333
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0008134, 0.0008248
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004935, 0.0004936
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019631, 0.0019649
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001284, 0.0001296
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0048031, 0.0047920
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0051225, 0.0051257
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0024191, 0.0024156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003576, upper bound: 0.0003767
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003634, upper bound: 0.0003717
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025819, 0.0025851
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004588, 0.0004573
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0032449, 0.0032488
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0008152, 0.0008249
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004959, 0.0004956
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019710, 0.0019735
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001284, 0.0001297
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0048098, 0.0048010
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0051483, 0.0051470
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0024290, 0.0024280

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003658, upper bound: 0.0003756
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003663, upper bound: 0.0003754
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025852, 0.0025828
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004600, 0.0004563
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0032502, 0.0032453
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0008133, 0.0008274
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004952, 0.0004965
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019735, 0.0019717
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001284, 0.0001298
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0048086, 0.0048033
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0051401, 0.0051571
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0024339, 0.0024238

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003477, upper bound: 0.0003692
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003534, upper bound: 0.0003615
time: 0.94 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.35 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.35
Output dim: 3, lower bound: -0.0003401, upper bound: 0.0003469
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.35
Output dim: 3, lower bound: -0.0003467, upper bound: 0.0003407
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -0.0003657, upper bound: 0.0003489
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -0.0003630, upper bound: 0.0003512
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.35
Output dim: 3, lower bound: -0.0003481, upper bound: 0.0003419
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.35
Output dim: 3, lower bound: -0.0003473, upper bound: 0.0003429
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -0.0003637, upper bound: 0.0003612
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -0.0003595, upper bound: 0.0003619
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -0.0003617, upper bound: 0.0003417
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -0.0003604, upper bound: 0.0003427
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -0.0003771, upper bound: 0.0003552
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -0.0003752, upper bound: 0.0003573
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.35
Output dim: 3, lower bound: -0.0003160, upper bound: 0.0003215
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.35
Output dim: 3, lower bound: -0.0003160, upper bound: 0.0003214
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -0.0003596, upper bound: 0.0003702
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -0.0003596, upper bound: 0.0003702
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -0.0003633, upper bound: 0.0003614
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -0.0003624, upper bound: 0.0003630
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.35
Output dim: 3, lower bound: -0.0003412, upper bound: 0.0003514
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.35
Output dim: 3, lower bound: -0.0003478, upper bound: 0.0003463
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.35
Output dim: 3, lower bound: -0.0003336, upper bound: 0.0003472
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.35
Output dim: 3, lower bound: -0.0003327, upper bound: 0.0003485
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -0.0003378, upper bound: 0.0003612
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -0.0003458, upper bound: 0.0003566
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.35
Output dim: 3, lower bound: -0.0003308, upper bound: 0.0003472
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.35
Output dim: 3, lower bound: -0.0003389, upper bound: 0.0003422
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -0.0003524, upper bound: 0.0003627
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -0.0003512, upper bound: 0.0003635
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -0.0003576, upper bound: 0.0003767
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -0.0003634, upper bound: 0.0003717
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -0.0003658, upper bound: 0.0003756
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -0.0003663, upper bound: 0.0003754
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -0.0003477, upper bound: 0.0003692
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -0.0003534, upper bound: 0.0003615

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025136, 0.0025179
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004321, 0.0004327
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031476, 0.0031506
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007524, 0.0007500
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004787, 0.0004784
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019180, 0.0019210
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001259, 0.0001263
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0047574, 0.0047396
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0049578, 0.0049542
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023315, 0.0023330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003610, upper bound: 0.0003432
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003583, upper bound: 0.0003446
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025156, 0.0025173
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004326, 0.0004323
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031500, 0.0031500
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007517, 0.0007514
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004786, 0.0004787
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019196, 0.0019206
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001258, 0.0001263
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0047565, 0.0047411
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0049562, 0.0049578
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023333, 0.0023319

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003585, upper bound: 0.0003452
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003566, upper bound: 0.0003468
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0024896, 0.0024878
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004305, 0.0004293
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031173, 0.0031138
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007862, 0.0007895
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004733, 0.0004739
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0018996, 0.0018981
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001283, 0.0001286
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0047147, 0.0047071
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0049017, 0.0049078
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023086, 0.0023052

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003592, upper bound: 0.0003546
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003562, upper bound: 0.0003567
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0024902, 0.0024869
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004308, 0.0004290
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031179, 0.0031128
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007851, 0.0007901
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004731, 0.0004740
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019001, 0.0018974
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001283, 0.0001286
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0047136, 0.0047072
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0049000, 0.0049093
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023094, 0.0023044

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003404, upper bound: 0.0003482
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003453, upper bound: 0.0003423
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0024662, 0.0024662
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004305, 0.0004323
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0030899, 0.0030907
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007876, 0.0007841
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004706, 0.0004703
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0018819, 0.0018819
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001271, 0.0001264
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0046619, 0.0046702
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0048784, 0.0048727
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0022953, 0.0022990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003537, upper bound: 0.0003335
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003525, upper bound: 0.0003344
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0024580, 0.0024745
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004303, 0.0004324
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0030811, 0.0030997
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007874, 0.0007842
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004718, 0.0004692
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0018758, 0.0018882
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001260, 0.0001275
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0046863, 0.0046416
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0048874, 0.0048635
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0022919, 0.0023021

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003386, upper bound: 0.0003278
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003469, upper bound: 0.0003227
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025019, 0.0025092
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004328, 0.0004351
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031323, 0.0031410
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007884, 0.0007848
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004776, 0.0004762
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019090, 0.0019145
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001292, 0.0001296
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0047572, 0.0047396
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0049485, 0.0049321
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023221, 0.0023308

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002344, upper bound: 0.0002335
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002344, upper bound: 0.0002335
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025031, 0.0025087
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004331, 0.0004348
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031337, 0.0031400
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007877, 0.0007857
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004773, 0.0004764
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019099, 0.0019140
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001291, 0.0001296
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0047568, 0.0047403
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0049450, 0.0049346
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023234, 0.0023289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003750, upper bound: 0.0003567
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003735, upper bound: 0.0003572
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0024365, 0.0024198
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004435, 0.0004416
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0030686, 0.0030469
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0008268, 0.0008229
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004663, 0.0004696
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0018605, 0.0018476
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001240, 0.0001229
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0044917, 0.0045245
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0048511, 0.0048836
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023105, 0.0022962

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003466, upper bound: 0.0003630
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003525, upper bound: 0.0003589
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0024364, 0.0024181
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004435, 0.0004409
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0030677, 0.0030451
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0008258, 0.0008233
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004660, 0.0004693
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0018604, 0.0018465
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001240, 0.0001229
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0044896, 0.0045239
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0048473, 0.0048817
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023098, 0.0022939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003467, upper bound: 0.0003630
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003525, upper bound: 0.0003589
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0023767, 0.0023911
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004199, 0.0004244
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0029800, 0.0030000
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007974, 0.0007887
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004568, 0.0004535
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0018137, 0.0018249
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001261, 0.0001254
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0045114, 0.0045056
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0047352, 0.0047005
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0022165, 0.0022339

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003590, upper bound: 0.0003560
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003570, upper bound: 0.0003574
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0023821, 0.0023857
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004218, 0.0004226
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0029881, 0.0029922
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007967, 0.0007898
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004555, 0.0004551
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0018179, 0.0018207
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001259, 0.0001260
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0045109, 0.0045086
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0047209, 0.0047187
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0022257, 0.0022264

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003614, upper bound: 0.0003576
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003607, upper bound: 0.0003620
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0022962, 0.0022544
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0003895, 0.0003819
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0028643, 0.0028121
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007319, 0.0007425
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004254, 0.0004334
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0017510, 0.0017191
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001246, 0.0001240
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0043328, 0.0044123
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0043920, 0.0044752
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0020980, 0.0020587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003358, upper bound: 0.0003607
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003361, upper bound: 0.0003609
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0022958, 0.0022561
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0003886, 0.0003830
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0028633, 0.0028155
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007338, 0.0007415
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004260, 0.0004331
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0017507, 0.0017206
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001247, 0.0001239
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0043333, 0.0044126
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0043992, 0.0044711
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0020956, 0.0020626

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003414, upper bound: 0.0003487
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003392, upper bound: 0.0003520
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025310, 0.0025221
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004525, 0.0004493
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031827, 0.0031721
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0008095, 0.0008211
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004847, 0.0004863
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019322, 0.0019255
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001263, 0.0001265
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0046945, 0.0047145
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0050334, 0.0050501
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023840, 0.0023752

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003513, upper bound: 0.0003585
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003504, upper bound: 0.0003616
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025213, 0.0025306
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004523, 0.0004495
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031723, 0.0031804
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0008093, 0.0008213
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004856, 0.0004851
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019249, 0.0019318
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001252, 0.0001276
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0047220, 0.0046855
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0050401, 0.0050406
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023810, 0.0023776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003398, upper bound: 0.0003563
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003442, upper bound: 0.0003518
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025904, 0.0025923
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004578, 0.0004545
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0032572, 0.0032576
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007744, 0.0007863
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004973, 0.0004979
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019775, 0.0019789
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001251, 0.0001265
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0048208, 0.0048101
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0051629, 0.0051723
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0024404, 0.0024338

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003127, upper bound: 0.0003247
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003127, upper bound: 0.0003247
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025898, 0.0025939
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004569, 0.0004556
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0032557, 0.0032609
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007760, 0.0007855
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004980, 0.0004974
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019770, 0.0019802
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001252, 0.0001264
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0048209, 0.0048096
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0051701, 0.0051661
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0024372, 0.0024377

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003590, upper bound: 0.0003628
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003575, upper bound: 0.0003670
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025609, 0.0025634
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004551, 0.0004531
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0032159, 0.0032186
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0008117, 0.0008216
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004912, 0.0004911
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019547, 0.0019567
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001283, 0.0001296
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0047809, 0.0047734
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0050992, 0.0051007
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0024077, 0.0024053

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003462, upper bound: 0.0003621
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003514, upper bound: 0.0003549
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025600, 0.0025640
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004550, 0.0004536
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0032153, 0.0032198
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0008119, 0.0008213
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004914, 0.0004910
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019540, 0.0019572
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001283, 0.0001296
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0047822, 0.0047721
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0051021, 0.0051004
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0024075, 0.0024067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003531, upper bound: 0.0003680
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003591, upper bound: 0.0003628
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0022950, 0.0022496
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0003964, 0.0003882
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0028635, 0.0028072
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007689, 0.0007831
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004251, 0.0004337
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0017501, 0.0017157
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001279, 0.0001274
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0043367, 0.0044212
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0043954, 0.0044841
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0021065, 0.0020648

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003011, upper bound: 0.0003156
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003011, upper bound: 0.0003156
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0022519, 0.0022893
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0003918, 0.0003917
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0028121, 0.0028536
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007694, 0.0007829
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004317, 0.0004264
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0017174, 0.0017456
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001260, 0.0001292
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0044250, 0.0043297
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0044592, 0.0044120
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0020746, 0.0020924

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003524, upper bound: 0.0003582
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003514, upper bound: 0.0003605
time: 0.99 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.28 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003610, upper bound: 0.0003432
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003583, upper bound: 0.0003446
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003585, upper bound: 0.0003452
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003566, upper bound: 0.0003468
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003592, upper bound: 0.0003546
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003562, upper bound: 0.0003567
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003404, upper bound: 0.0003482
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003453, upper bound: 0.0003423
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003537, upper bound: 0.0003335
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003525, upper bound: 0.0003344
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003386, upper bound: 0.0003278
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003469, upper bound: 0.0003227
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0002344, upper bound: 0.0002335
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0002344, upper bound: 0.0002335
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003750, upper bound: 0.0003567
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003735, upper bound: 0.0003572
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003466, upper bound: 0.0003630
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003525, upper bound: 0.0003589
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003467, upper bound: 0.0003630
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003525, upper bound: 0.0003589
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003590, upper bound: 0.0003560
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003570, upper bound: 0.0003574
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003614, upper bound: 0.0003576
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003607, upper bound: 0.0003620
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003358, upper bound: 0.0003607
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003361, upper bound: 0.0003609
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003414, upper bound: 0.0003487
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003392, upper bound: 0.0003520
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003513, upper bound: 0.0003585
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003504, upper bound: 0.0003616
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003398, upper bound: 0.0003563
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003442, upper bound: 0.0003518
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003127, upper bound: 0.0003247
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003127, upper bound: 0.0003247
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003590, upper bound: 0.0003628
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003575, upper bound: 0.0003670
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003462, upper bound: 0.0003621
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003514, upper bound: 0.0003549
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003531, upper bound: 0.0003680
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003591, upper bound: 0.0003628
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003011, upper bound: 0.0003156
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003011, upper bound: 0.0003156
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003524, upper bound: 0.0003582
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 3, lower bound: -0.0003514, upper bound: 0.0003605

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025088, 0.0025135
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004310, 0.0004317
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031432, 0.0031466
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007476, 0.0007441
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004784, 0.0004781
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019144, 0.0019177
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001250, 0.0001253
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0047389, 0.0047206
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0049572, 0.0049535
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023306, 0.0023322

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003410, upper bound: 0.0003241
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003406, upper bound: 0.0003252
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025092, 0.0025131
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004312, 0.0004316
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031436, 0.0031462
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007464, 0.0007453
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004784, 0.0004781
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019147, 0.0019175
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001249, 0.0001255
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0047383, 0.0047213
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0049572, 0.0049536
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023308, 0.0023321

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003582, upper bound: 0.0003444
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003581, upper bound: 0.0003444
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025107, 0.0025129
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004315, 0.0004313
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031455, 0.0031460
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007470, 0.0007455
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004783, 0.0004784
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019159, 0.0019173
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001250, 0.0001254
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0047381, 0.0047216
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0049556, 0.0049571
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023324, 0.0023311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003390, upper bound: 0.0003262
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003385, upper bound: 0.0003267
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025112, 0.0025126
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004317, 0.0004312
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031460, 0.0031458
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007458, 0.0007467
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004783, 0.0004784
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019163, 0.0019171
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001249, 0.0001255
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0047381, 0.0047228
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0049556, 0.0049572
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023326, 0.0023310

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003522, upper bound: 0.0003407
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003487, upper bound: 0.0003423
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0024774, 0.0024789
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004281, 0.0004279
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031027, 0.0031038
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007839, 0.0007846
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004717, 0.0004715
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0018903, 0.0018914
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001283, 0.0001286
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0046946, 0.0046845
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0048858, 0.0048834
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0022965, 0.0022975

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003480, upper bound: 0.0003473
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003518, upper bound: 0.0003403
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0024807, 0.0024751
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004290, 0.0004268
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031073, 0.0030982
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007816, 0.0007872
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004708, 0.0004723
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0018929, 0.0018884
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001283, 0.0001286
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0046916, 0.0046869
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0048754, 0.0048919
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023008, 0.0022925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003559, upper bound: 0.0003556
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003550, upper bound: 0.0003563
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0023969, 0.0024097
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004218, 0.0004256
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0030051, 0.0030225
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007887, 0.0007860
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004604, 0.0004575
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0018291, 0.0018389
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001260, 0.0001260
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0045508, 0.0045326
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0047754, 0.0047435
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0022357, 0.0022516

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003310, upper bound: 0.0003163
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003311, upper bound: 0.0003158
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0024041, 0.0024040
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004237, 0.0004233
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0030161, 0.0030137
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007874, 0.0007868
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004588, 0.0004595
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0018349, 0.0018345
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001254, 0.0001264
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0045474, 0.0045329
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0047570, 0.0047649
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0022460, 0.0022415

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002330, upper bound: 0.0002335
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002330, upper bound: 0.0002335
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0024602, 0.0024423
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004436, 0.0004409
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031005, 0.0030776
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007880, 0.0007852
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004712, 0.0004747
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0018788, 0.0018651
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001203, 0.0001192
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0045132, 0.0045464
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0049008, 0.0049374
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023355, 0.0023185

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003298, upper bound: 0.0003530
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003364, upper bound: 0.0003466
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0024590, 0.0024437
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004429, 0.0004425
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0030993, 0.0030808
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007910, 0.0007842
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004719, 0.0004745
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0018780, 0.0018664
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001204, 0.0001192
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0045141, 0.0045460
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0049085, 0.0049333
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023328, 0.0023226

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003074, upper bound: 0.0003113
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003074, upper bound: 0.0003113
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0024600, 0.0024406
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004435, 0.0004402
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031001, 0.0030758
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007870, 0.0007850
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004709, 0.0004747
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0018787, 0.0018640
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001202, 0.0001192
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0045111, 0.0045454
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0048970, 0.0049364
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023354, 0.0023162

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003298, upper bound: 0.0003530
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003365, upper bound: 0.0003466
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0024589, 0.0024423
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004429, 0.0004418
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0030984, 0.0030779
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007905, 0.0007846
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004714, 0.0004742
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0018779, 0.0018652
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001205, 0.0001192
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0045118, 0.0045454
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0049042, 0.0049314
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023321, 0.0023212

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003074, upper bound: 0.0003113
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003074, upper bound: 0.0003113
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0023727, 0.0023870
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004188, 0.0004234
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0029765, 0.0029965
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007923, 0.0007825
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004565, 0.0004533
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0018108, 0.0018220
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001254, 0.0001245
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0044950, 0.0044891
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0047346, 0.0046999
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0022155, 0.0022331

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003392, upper bound: 0.0003417
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003451, upper bound: 0.0003366
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0023726, 0.0023867
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004188, 0.0004232
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0029765, 0.0029962
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007912, 0.0007835
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004565, 0.0004533
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0018108, 0.0018217
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001252, 0.0001246
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0044944, 0.0044892
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0047345, 0.0046999
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0022155, 0.0022330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003560, upper bound: 0.0003525
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003554, upper bound: 0.0003564
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0023832, 0.0023872
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004205, 0.0004217
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0029890, 0.0029932
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007959, 0.0007879
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004556, 0.0004552
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0018188, 0.0018219
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001259, 0.0001260
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0045097, 0.0045072
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0047211, 0.0047180
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0022232, 0.0022251

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002295, upper bound: 0.0002286
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002295, upper bound: 0.0002286
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0023833, 0.0023868
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004209, 0.0004212
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0029893, 0.0029930
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007948, 0.0007891
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004556, 0.0004553
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0018188, 0.0018216
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001259, 0.0001260
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0045095, 0.0045062
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0047202, 0.0047192
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0022240, 0.0022240

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003459, upper bound: 0.0003546
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003535, upper bound: 0.0003513
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0021961, 0.0021555
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0003798, 0.0003733
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0027417, 0.0026920
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007353, 0.0007448
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004080, 0.0004154
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0016749, 0.0016440
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001211, 0.0001200
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0041569, 0.0042400
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0042197, 0.0042948
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0020164, 0.0019817

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003374, upper bound: 0.0003602
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003374, upper bound: 0.0003606
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0021972, 0.0021484
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0003809, 0.0003709
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0027441, 0.0026811
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007342, 0.0007457
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004061, 0.0004160
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0016758, 0.0016384
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001205, 0.0001205
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0041573, 0.0042358
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0041983, 0.0043025
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0020208, 0.0019707

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002923, upper bound: 0.0003094
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002923, upper bound: 0.0003093
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025308, 0.0025230
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004504, 0.0004476
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031817, 0.0031724
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0008098, 0.0008203
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004846, 0.0004859
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019322, 0.0019263
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001263, 0.0001264
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0046875, 0.0047069
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0050316, 0.0050458
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023811, 0.0023737

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003510, upper bound: 0.0003567
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003505, upper bound: 0.0003583
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025322, 0.0025219
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004509, 0.0004472
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031837, 0.0031711
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0008088, 0.0008214
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004843, 0.0004863
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019332, 0.0019255
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001263, 0.0001265
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0046873, 0.0047067
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0050291, 0.0050503
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023836, 0.0023724

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003501, upper bound: 0.0003596
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003496, upper bound: 0.0003614
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025394, 0.0025474
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004526, 0.0004490
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031951, 0.0032012
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007701, 0.0007832
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004889, 0.0004889
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019388, 0.0019446
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001219, 0.0001244
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0047397, 0.0047036
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0050773, 0.0050831
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0024007, 0.0023946

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002843, upper bound: 0.0002949
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002843, upper bound: 0.0002949
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025760, 0.0025833
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004540, 0.0004539
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0032376, 0.0032472
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007753, 0.0007824
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004958, 0.0004944
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019665, 0.0019721
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001251, 0.0001264
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0048011, 0.0047878
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0051468, 0.0051335
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0024216, 0.0024268

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003403, upper bound: 0.0003530
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003495, upper bound: 0.0003489
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025792, 0.0025800
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004552, 0.0004529
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0032420, 0.0032417
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007729, 0.0007847
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004948, 0.0004952
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019690, 0.0019695
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001251, 0.0001264
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0047998, 0.0047895
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0051366, 0.0051428
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0024265, 0.0024217

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003374, upper bound: 0.0003544
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003434, upper bound: 0.0003444
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0022702, 0.0022309
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0003904, 0.0003843
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0028301, 0.0027826
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007671, 0.0007772
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004211, 0.0004281
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0017311, 0.0017013
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001278, 0.0001273
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0043112, 0.0043919
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0043531, 0.0044226
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0020748, 0.0020428

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003017, upper bound: 0.0003100
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003017, upper bound: 0.0003100
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025780, 0.0025808
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004551, 0.0004530
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0032409, 0.0032430
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007736, 0.0007832
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004950, 0.0004951
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019680, 0.0019702
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001251, 0.0001265
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0048002, 0.0047909
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0051387, 0.0051417
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0024260, 0.0024226

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003521, upper bound: 0.0003650
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003514, upper bound: 0.0003670
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025768, 0.0025821
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004544, 0.0004539
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0032385, 0.0032446
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007754, 0.0007827
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004954, 0.0004946
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019670, 0.0019711
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001251, 0.0001264
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0048002, 0.0047902
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0051439, 0.0051371
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0024235, 0.0024256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003403, upper bound: 0.0003530
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003495, upper bound: 0.0003491
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0022513, 0.0022895
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0003909, 0.0003914
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0028113, 0.0028546
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007692, 0.0007817
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004319, 0.0004263
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0017170, 0.0017459
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001260, 0.0001291
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0044210, 0.0043260
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0044632, 0.0044122
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0020744, 0.0020944

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003522, upper bound: 0.0003558
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003519, upper bound: 0.0003579
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0022523, 0.0022887
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0003912, 0.0003908
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0028125, 0.0028529
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007682, 0.0007828
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004316, 0.0004265
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0017178, 0.0017453
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001260, 0.0001292
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0044219, 0.0043272
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0044595, 0.0044138
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0020753, 0.0020923

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003304, upper bound: 0.0003506
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003425, upper bound: 0.0003450
time: 0.98 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.34 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003410, upper bound: 0.0003241
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003406, upper bound: 0.0003252
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003582, upper bound: 0.0003444
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003581, upper bound: 0.0003444
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003390, upper bound: 0.0003262
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003385, upper bound: 0.0003267
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003522, upper bound: 0.0003407
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003487, upper bound: 0.0003423
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003480, upper bound: 0.0003473
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003518, upper bound: 0.0003403
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003559, upper bound: 0.0003556
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003550, upper bound: 0.0003563
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003310, upper bound: 0.0003163
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003311, upper bound: 0.0003158
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0002330, upper bound: 0.0002335
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0002330, upper bound: 0.0002335
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003298, upper bound: 0.0003530
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003364, upper bound: 0.0003466
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003074, upper bound: 0.0003113
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003074, upper bound: 0.0003113
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003298, upper bound: 0.0003530
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003365, upper bound: 0.0003466
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003074, upper bound: 0.0003113
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003074, upper bound: 0.0003113
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003392, upper bound: 0.0003417
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003451, upper bound: 0.0003366
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003560, upper bound: 0.0003525
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003554, upper bound: 0.0003564
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0002295, upper bound: 0.0002286
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0002295, upper bound: 0.0002286
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003459, upper bound: 0.0003546
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003535, upper bound: 0.0003513
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003374, upper bound: 0.0003602
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003374, upper bound: 0.0003606
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0002923, upper bound: 0.0003094
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0002923, upper bound: 0.0003093
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003510, upper bound: 0.0003567
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003505, upper bound: 0.0003583
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003501, upper bound: 0.0003596
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003496, upper bound: 0.0003614
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0002843, upper bound: 0.0002949
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0002843, upper bound: 0.0002949
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003403, upper bound: 0.0003530
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003495, upper bound: 0.0003489
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003374, upper bound: 0.0003544
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003434, upper bound: 0.0003444
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003017, upper bound: 0.0003100
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003017, upper bound: 0.0003100
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003521, upper bound: 0.0003650
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003514, upper bound: 0.0003670
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003403, upper bound: 0.0003530
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003495, upper bound: 0.0003491
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003522, upper bound: 0.0003558
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003519, upper bound: 0.0003579
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003304, upper bound: 0.0003506
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 3, lower bound: -0.0003425, upper bound: 0.0003450

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025205, 0.0025257
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004343, 0.0004348
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031579, 0.0031632
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007410, 0.0007400
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004812, 0.0004806
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019234, 0.0019272
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001250, 0.0001255
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0047570, 0.0047395
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0049882, 0.0049819
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023451, 0.0023476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003397, upper bound: 0.0003247
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003386, upper bound: 0.0003255
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025218, 0.0025246
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004344, 0.0004348
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031605, 0.0031631
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007407, 0.0007400
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004813, 0.0004809
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019245, 0.0019265
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001250, 0.0001256
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0047545, 0.0047407
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0049886, 0.0049848
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023464, 0.0023480

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003393, upper bound: 0.0003246
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003385, upper bound: 0.0003254
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0023813, 0.0023816
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004199, 0.0004198
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0029878, 0.0029870
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007831, 0.0007874
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004551, 0.0004552
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0018175, 0.0018176
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001251, 0.0001249
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0044983, 0.0044909
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0047191, 0.0047202
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0022238, 0.0022233

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003086, upper bound: 0.0003076
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003086, upper bound: 0.0003076
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0023871, 0.0023765
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004220, 0.0004180
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0029961, 0.0029794
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007821, 0.0007886
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004536, 0.0004567
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0018221, 0.0018135
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001246, 0.0001255
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0044975, 0.0044922
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0047018, 0.0047355
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0022315, 0.0022143

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003372, upper bound: 0.0003364
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003361, upper bound: 0.0003378
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0023738, 0.0023885
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004175, 0.0004224
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0029774, 0.0029979
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007904, 0.0007816
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004568, 0.0004534
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0018117, 0.0018231
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001252, 0.0001246
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0044935, 0.0044876
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0047360, 0.0046992
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0022131, 0.0022319

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003423, upper bound: 0.0003452
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003487, upper bound: 0.0003420
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0023741, 0.0023879
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004180, 0.0004219
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0029780, 0.0029971
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007893, 0.0007829
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004566, 0.0004536
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0018118, 0.0018227
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001252, 0.0001246
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0044929, 0.0044860
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0047339, 0.0047014
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0022144, 0.0022306

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003416, upper bound: 0.0003490
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003482, upper bound: 0.0003460
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0022087, 0.0021716
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0003778, 0.0003723
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0027543, 0.0027093
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007299, 0.0007393
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004106, 0.0004173
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0016842, 0.0016560
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001214, 0.0001202
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0041861, 0.0042674
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0042476, 0.0043155
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0020256, 0.0019948

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003193, upper bound: 0.0003383
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003192, upper bound: 0.0003401
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0022122, 0.0021705
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0003787, 0.0003724
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0027591, 0.0027082
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007299, 0.0007397
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004105, 0.0004180
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0016869, 0.0016552
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001213, 0.0001202
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0041853, 0.0042692
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0042467, 0.0043230
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0020297, 0.0019945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003176, upper bound: 0.0003385
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003192, upper bound: 0.0003406
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0024201, 0.0024252
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004396, 0.0004397
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0030470, 0.0030559
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0008123, 0.0008218
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004680, 0.0004664
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0018478, 0.0018520
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001222, 0.0001223
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0044918, 0.0044943
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0048694, 0.0048528
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0022951, 0.0023021

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003315, upper bound: 0.0003435
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003367, upper bound: 0.0003348
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0024330, 0.0024184
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004425, 0.0004379
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0030653, 0.0030453
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0008115, 0.0008228
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004661, 0.0004694
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0018579, 0.0018466
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001221, 0.0001229
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0044929, 0.0045100
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0048481, 0.0048835
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023095, 0.0022918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003304, upper bound: 0.0003446
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003364, upper bound: 0.0003372
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0024206, 0.0024242
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004400, 0.0004394
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0030475, 0.0030546
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0008113, 0.0008228
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004678, 0.0004665
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0018483, 0.0018512
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001221, 0.0001223
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0044916, 0.0044939
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0048669, 0.0048544
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0022960, 0.0023008

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003462, upper bound: 0.0003527
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003447, upper bound: 0.0003554
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0024344, 0.0024179
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004430, 0.0004373
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0030672, 0.0030440
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0008103, 0.0008239
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004659, 0.0004698
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0018590, 0.0018461
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001220, 0.0001229
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0044926, 0.0045098
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0048455, 0.0048880
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023119, 0.0022903

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003311, upper bound: 0.0003514
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003402, upper bound: 0.0003457
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025780, 0.0025814
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004537, 0.0004519
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0032407, 0.0032434
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007711, 0.0007796
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004950, 0.0004948
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019680, 0.0019706
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001251, 0.0001264
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0047959, 0.0047874
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0051386, 0.0051391
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0024241, 0.0024222

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003349, upper bound: 0.0003450
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003348, upper bound: 0.0003459
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0025793, 0.0025809
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004543, 0.0004516
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0032426, 0.0032428
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007701, 0.0007810
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004948, 0.0004951
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019690, 0.0019702
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001250, 0.0001264
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0047967, 0.0047874
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0051361, 0.0051428
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0024260, 0.0024208

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003487, upper bound: 0.0003652
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003503, upper bound: 0.0003666
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0021686, 0.0022141
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0003807, 0.0003840
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0027086, 0.0027630
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007721, 0.0007832
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004187, 0.0004111
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0016539, 0.0016885
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001230, 0.0001257
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0042909, 0.0041930
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0043304, 0.0042568
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0020030, 0.0020345

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003325, upper bound: 0.0003461
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003429, upper bound: 0.0003391
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0021759, 0.0022143
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0003833, 0.0003825
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0027197, 0.0027619
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007712, 0.0007844
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004183, 0.0004131
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0016596, 0.0016886
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001225, 0.0001263
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0043002, 0.0041957
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0043241, 0.0042792
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0020144, 0.0020304

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003034, upper bound: 0.0003087
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003034, upper bound: 0.0003087
time: 1.02 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 3.33 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 3, lower bound: -0.0003397, upper bound: 0.0003247
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 3, lower bound: -0.0003386, upper bound: 0.0003255
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 3, lower bound: -0.0003393, upper bound: 0.0003246
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 3, lower bound: -0.0003385, upper bound: 0.0003254
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 3, lower bound: -0.0003086, upper bound: 0.0003076
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 3, lower bound: -0.0003086, upper bound: 0.0003076
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 3, lower bound: -0.0003372, upper bound: 0.0003364
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 3, lower bound: -0.0003361, upper bound: 0.0003378
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 3, lower bound: -0.0003423, upper bound: 0.0003452
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 3, lower bound: -0.0003487, upper bound: 0.0003420
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 3, lower bound: -0.0003416, upper bound: 0.0003490
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 3, lower bound: -0.0003482, upper bound: 0.0003460
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 3, lower bound: -0.0003193, upper bound: 0.0003383
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 3, lower bound: -0.0003192, upper bound: 0.0003401
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 3, lower bound: -0.0003176, upper bound: 0.0003385
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 3, lower bound: -0.0003192, upper bound: 0.0003406
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 3, lower bound: -0.0003315, upper bound: 0.0003435
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 3, lower bound: -0.0003367, upper bound: 0.0003348
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 3, lower bound: -0.0003304, upper bound: 0.0003446
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 3, lower bound: -0.0003364, upper bound: 0.0003372
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 3, lower bound: -0.0003462, upper bound: 0.0003527
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 3, lower bound: -0.0003447, upper bound: 0.0003554
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 3, lower bound: -0.0003311, upper bound: 0.0003514
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 3, lower bound: -0.0003402, upper bound: 0.0003457
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 3, lower bound: -0.0003349, upper bound: 0.0003450
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 3, lower bound: -0.0003348, upper bound: 0.0003459
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.33
Output dim: 3, lower bound: -0.0003487, upper bound: 0.0003652
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.33
Output dim: 3, lower bound: -0.0003503, upper bound: 0.0003666
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 3, lower bound: -0.0003325, upper bound: 0.0003461
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 3, lower bound: -0.0003429, upper bound: 0.0003391
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 3, lower bound: -0.0003034, upper bound: 0.0003087
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 3, lower bound: -0.0003034, upper bound: 0.0003087

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0024916, 0.0024989
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004458, 0.0004452
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031402, 0.0031492
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007721, 0.0007819
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004821, 0.0004809
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019028, 0.0019085
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001210, 0.0001221
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0046264, 0.0046149
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0050118, 0.0050006
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023628, 0.0023666

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003059, upper bound: 0.0003150
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003059, upper bound: 0.0003150
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0024972, 0.0024918
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004478, 0.0004428
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0031490, 0.0031380
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007711, 0.0007830
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004800, 0.0004824
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0019073, 0.0019029
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001206, 0.0001226
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0046270, 0.0046154
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0049889, 0.0050184
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0023716, 0.0023553

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003319, upper bound: 0.0003569
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003409, upper bound: 0.0003514
time: 1.02 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 3.40 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 3.40
Output dim: 3, lower bound: -0.0003059, upper bound: 0.0003150
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 3.40
Output dim: 3, lower bound: -0.0003059, upper bound: 0.0003150
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 3.40
Output dim: 3, lower bound: -0.0003319, upper bound: 0.0003569
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 3.40
Output dim: 3, lower bound: -0.0003409, upper bound: 0.0003514

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044699, -0.0014030, -0.0044699, -0.0014030, -0.0024333, 0.0024223
1: -0.0047143, -0.0041600, -0.0047143, -0.0041600, -0.0004270, 0.0004203
2: 0.0088821, 0.0127581, 0.0088821, 0.0127581, -0.0030574, 0.0030394
3: 1.0083551, 1.0093288, 1.0083551, 1.0093288, -0.0007306, 0.0007453
4: -0.0036802, -0.0030839, -0.0036802, -0.0030839, -0.0004630, 0.0004666
5: 0.0005307, 0.0028738, 0.0005307, 0.0028738, -0.0018576, 0.0018489
6: -0.0025678, -0.0024298, -0.0025678, -0.0024298, -0.0001204, 0.0001224
7: -0.0114657, -0.0059010, -0.0114657, -0.0059010, -0.0045495, 0.0045432
8: -0.0072755, -0.0010605, -0.0072755, -0.0010605, -0.0048013, 0.0048445
9: -0.0036278, -0.0006921, -0.0036278, -0.0006921, -0.0022836, 0.0022602

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=246
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003116, upper bound: 0.0003434
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003179, upper bound: 0.0003362
time: 1.01 seconds

## Summary of splitting (split count: 10)
- Time for RS candidates: 3.32 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 3.32
Output dim: 3, lower bound: -0.0003116, upper bound: 0.0003434
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 3.32
Output dim: 3, lower bound: -0.0003179, upper bound: 0.0003362

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.22 + 436.46 = 439.68 seconds
