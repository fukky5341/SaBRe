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
Threshold: 4.776e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0035298, -0.0028553, -0.0035298, -0.0028553, -0.0002382, 0.0002382)
1: (-0.0045334, -0.0044117, -0.0045334, -0.0044117, -0.0000422, 0.0000422)
2: (0.0100759, 0.0109296, 0.0100759, 0.0109296, -0.0003002, 0.0003002)
3: (1.0087553, 1.0089401, 1.0087553, 1.0089401, -0.0000734, 0.0000734)
4: (-0.0034002, -0.0032689, -0.0034002, -0.0032689, -0.0000460, 0.0000460)
5: (0.0012493, 0.0017647, 0.0012493, 0.0017647, -0.0001818, 0.0001818)
6: (-0.0025229, -0.0024961, -0.0025229, -0.0024961, -0.0000106, 0.0000106)
7: (-0.0087948, -0.0075867, -0.0087948, -0.0075867, -0.0004396, 0.0004396)
8: (-0.0043657, -0.0029997, -0.0043657, -0.0029997, -0.0004772, 0.0004772)
9: (-0.0027046, -0.0020597, -0.0027046, -0.0020597, -0.0002246, 0.0002246)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.46 + 1.26 = 2.72 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0000538, upper bound: 0.0000538

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000535, upper bound: 0.0000523
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000523, upper bound: 0.0000535
time: 0.43 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.00 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.00
Output dim: 3, lower bound: -0.0000535, upper bound: 0.0000523
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.00
Output dim: 3, lower bound: -0.0000523, upper bound: 0.0000535

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0035298, -0.0028553, -0.0035298, -0.0028553, -0.0002100, 0.0002094
1: -0.0045334, -0.0044117, -0.0045334, -0.0044117, -0.0000398, 0.0000399
2: 0.0100759, 0.0109296, 0.0100759, 0.0109296, -0.0002670, 0.0002662
3: 1.0087553, 1.0089401, 1.0087553, 1.0089401, -0.0000716, 0.0000713
4: -0.0034002, -0.0032689, -0.0034002, -0.0032689, -0.0000411, 0.0000413
5: 0.0012493, 0.0017647, 0.0012493, 0.0017647, -0.0001606, 0.0001600
6: -0.0025229, -0.0024961, -0.0025229, -0.0024961, -0.0000086, 0.0000086
7: -0.0087948, -0.0075867, -0.0087948, -0.0075867, -0.0003743, 0.0003750
8: -0.0043657, -0.0029997, -0.0043657, -0.0029997, -0.0004299, 0.0004311
9: -0.0027046, -0.0020597, -0.0027046, -0.0020597, -0.0002040, 0.0002038

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000511, upper bound: 0.0000490
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000501, upper bound: 0.0000502
time: 0.44 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0035298, -0.0028553, -0.0035298, -0.0028553, -0.0002094, 0.0002100
1: -0.0045334, -0.0044117, -0.0045334, -0.0044117, -0.0000399, 0.0000398
2: 0.0100759, 0.0109296, 0.0100759, 0.0109296, -0.0002662, 0.0002670
3: 1.0087553, 1.0089401, 1.0087553, 1.0089401, -0.0000713, 0.0000716
4: -0.0034002, -0.0032689, -0.0034002, -0.0032689, -0.0000413, 0.0000411
5: 0.0012493, 0.0017647, 0.0012493, 0.0017647, -0.0001600, 0.0001606
6: -0.0025229, -0.0024961, -0.0025229, -0.0024961, -0.0000086, 0.0000086
7: -0.0087948, -0.0075867, -0.0087948, -0.0075867, -0.0003750, 0.0003743
8: -0.0043657, -0.0029997, -0.0043657, -0.0029997, -0.0004311, 0.0004299
9: -0.0027046, -0.0020597, -0.0027046, -0.0020597, -0.0002038, 0.0002040

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000502, upper bound: 0.0000501
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000490, upper bound: 0.0000511
time: 0.45 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.47 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.47
Output dim: 3, lower bound: -0.0000511, upper bound: 0.0000490
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.47
Output dim: 3, lower bound: -0.0000501, upper bound: 0.0000502
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.47
Output dim: 3, lower bound: -0.0000502, upper bound: 0.0000501
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.47
Output dim: 3, lower bound: -0.0000490, upper bound: 0.0000511

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0035298, -0.0028553, -0.0035298, -0.0028553, -0.0002027, 0.0002000
1: -0.0045334, -0.0044117, -0.0045334, -0.0044117, -0.0000397, 0.0000398
2: 0.0100759, 0.0109296, 0.0100759, 0.0109296, -0.0002587, 0.0002556
3: 1.0087553, 1.0089401, 1.0087553, 1.0089401, -0.0000685, 0.0000671
4: -0.0034002, -0.0032689, -0.0034002, -0.0032689, -0.0000398, 0.0000401
5: 0.0012493, 0.0017647, 0.0012493, 0.0017647, -0.0001551, 0.0001530
6: -0.0025229, -0.0024961, -0.0025229, -0.0024961, -0.0000078, 0.0000074
7: -0.0087948, -0.0075867, -0.0087948, -0.0075867, -0.0003484, 0.0003539
8: -0.0043657, -0.0029997, -0.0043657, -0.0029997, -0.0004172, 0.0004199
9: -0.0027046, -0.0020597, -0.0027046, -0.0020597, -0.0001993, 0.0001989

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000457, upper bound: 0.0000458
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000480, upper bound: 0.0000429
time: 0.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0035298, -0.0028553, -0.0035298, -0.0028553, -0.0002007, 0.0002094
1: -0.0045334, -0.0044117, -0.0045334, -0.0044117, -0.0000397, 0.0000399
2: 0.0100759, 0.0109296, 0.0100759, 0.0109296, -0.0002565, 0.0002662
3: 1.0087553, 1.0089401, 1.0087553, 1.0089401, -0.0000674, 0.0000713
4: -0.0034002, -0.0032689, -0.0034002, -0.0032689, -0.0000411, 0.0000399
5: 0.0012493, 0.0017647, 0.0012493, 0.0017647, -0.0001535, 0.0001600
6: -0.0025229, -0.0024961, -0.0025229, -0.0024961, -0.0000075, 0.0000086
7: -0.0087948, -0.0075867, -0.0087948, -0.0075867, -0.0003743, 0.0003491
8: -0.0043657, -0.0029997, -0.0043657, -0.0029997, -0.0004299, 0.0004185
9: -0.0027046, -0.0020597, -0.0027046, -0.0020597, -0.0001991, 0.0002038

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000441, upper bound: 0.0000473
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000470, upper bound: 0.0000445
time: 0.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0035298, -0.0028553, -0.0035298, -0.0028553, -0.0002025, 0.0002007
1: -0.0045334, -0.0044117, -0.0045334, -0.0044117, -0.0000398, 0.0000397
2: 0.0100759, 0.0109296, 0.0100759, 0.0109296, -0.0002580, 0.0002565
3: 1.0087553, 1.0089401, 1.0087553, 1.0089401, -0.0000683, 0.0000674
4: -0.0034002, -0.0032689, -0.0034002, -0.0032689, -0.0000399, 0.0000400
5: 0.0012493, 0.0017647, 0.0012493, 0.0017647, -0.0001549, 0.0001535
6: -0.0025229, -0.0024961, -0.0025229, -0.0024961, -0.0000078, 0.0000075
7: -0.0087948, -0.0075867, -0.0087948, -0.0075867, -0.0003491, 0.0003525
8: -0.0043657, -0.0029997, -0.0043657, -0.0029997, -0.0004185, 0.0004198
9: -0.0027046, -0.0020597, -0.0027046, -0.0020597, -0.0001997, 0.0001991

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000445, upper bound: 0.0000470
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000472, upper bound: 0.0000441
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0035298, -0.0028553, -0.0035298, -0.0028553, -0.0002000, 0.0002100
1: -0.0045334, -0.0044117, -0.0045334, -0.0044117, -0.0000398, 0.0000398
2: 0.0100759, 0.0109296, 0.0100759, 0.0109296, -0.0002556, 0.0002670
3: 1.0087553, 1.0089401, 1.0087553, 1.0089401, -0.0000671, 0.0000716
4: -0.0034002, -0.0032689, -0.0034002, -0.0032689, -0.0000413, 0.0000398
5: 0.0012493, 0.0017647, 0.0012493, 0.0017647, -0.0001530, 0.0001606
6: -0.0025229, -0.0024961, -0.0025229, -0.0024961, -0.0000074, 0.0000086
7: -0.0087948, -0.0075867, -0.0087948, -0.0075867, -0.0003750, 0.0003484
8: -0.0043657, -0.0029997, -0.0043657, -0.0029997, -0.0004311, 0.0004172
9: -0.0027046, -0.0020597, -0.0027046, -0.0020597, -0.0001989, 0.0002040

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000429, upper bound: 0.0000480
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000458, upper bound: 0.0000457
time: 0.47 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.44 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.44
Output dim: 3, lower bound: -0.0000457, upper bound: 0.0000458
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 3, lower bound: -0.0000480, upper bound: 0.0000429
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.44
Output dim: 3, lower bound: -0.0000441, upper bound: 0.0000473
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.44
Output dim: 3, lower bound: -0.0000470, upper bound: 0.0000445
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.44
Output dim: 3, lower bound: -0.0000445, upper bound: 0.0000470
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.44
Output dim: 3, lower bound: -0.0000472, upper bound: 0.0000441
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 3, lower bound: -0.0000429, upper bound: 0.0000480
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.44
Output dim: 3, lower bound: -0.0000458, upper bound: 0.0000457

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0035298, -0.0028553, -0.0035298, -0.0028553, -0.0001690, 0.0001728
1: -0.0045334, -0.0044117, -0.0045334, -0.0044117, -0.0000315, 0.0000329
2: 0.0100759, 0.0109296, 0.0100759, 0.0109296, -0.0002133, 0.0002188
3: 1.0087553, 1.0089401, 1.0087553, 1.0089401, -0.0000584, 0.0000555
4: -0.0034002, -0.0032689, -0.0034002, -0.0032689, -0.0000337, 0.0000327
5: 0.0012493, 0.0017647, 0.0012493, 0.0017647, -0.0001291, 0.0001320
6: -0.0025229, -0.0024961, -0.0025229, -0.0024961, -0.0000076, 0.0000073
7: -0.0087948, -0.0075867, -0.0087948, -0.0075867, -0.0003119, 0.0003103
8: -0.0043657, -0.0029997, -0.0043657, -0.0029997, -0.0003514, 0.0003396
9: -0.0027046, -0.0020597, -0.0027046, -0.0020597, -0.0001599, 0.0001664

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.27 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000398, upper bound: 0.0000348
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000395, upper bound: 0.0000350
time: 0.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0035298, -0.0028553, -0.0035298, -0.0028553, -0.0001728, 0.0001798
1: -0.0045334, -0.0044117, -0.0045334, -0.0044117, -0.0000329, 0.0000316
2: 0.0100759, 0.0109296, 0.0100759, 0.0109296, -0.0002188, 0.0002255
3: 1.0087553, 1.0089401, 1.0087553, 1.0089401, -0.0000555, 0.0000614
4: -0.0034002, -0.0032689, -0.0034002, -0.0032689, -0.0000343, 0.0000337
5: 0.0012493, 0.0017647, 0.0012493, 0.0017647, -0.0001320, 0.0001372
6: -0.0025229, -0.0024961, -0.0025229, -0.0024961, -0.0000073, 0.0000085
7: -0.0087948, -0.0075867, -0.0087948, -0.0075867, -0.0003377, 0.0003119
8: -0.0043657, -0.0029997, -0.0043657, -0.0029997, -0.0003549, 0.0003514
9: -0.0027046, -0.0020597, -0.0027046, -0.0020597, -0.0001664, 0.0001661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.26 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000350, upper bound: 0.0000395
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000348, upper bound: 0.0000398
time: 0.47 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.51 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.51
Output dim: 3, lower bound: -0.0000398, upper bound: 0.0000348
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.51
Output dim: 3, lower bound: -0.0000395, upper bound: 0.0000350
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.51
Output dim: 3, lower bound: -0.0000350, upper bound: 0.0000395
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.51
Output dim: 3, lower bound: -0.0000348, upper bound: 0.0000398

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.72 + 20.49 = 23.21 seconds
