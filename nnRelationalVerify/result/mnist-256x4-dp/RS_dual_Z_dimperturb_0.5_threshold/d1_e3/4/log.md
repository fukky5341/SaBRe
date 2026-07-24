## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00399952


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0153504, 0.0180217, 0.0153504, 0.0180217, -0.0021580, 0.0021580)
1: (-0.0017504, 0.0001848, -0.0017504, 0.0001848, -0.0015797, 0.0015797)
2: (0.0036439, 0.0045030, 0.0036439, 0.0045030, -0.0006909, 0.0006909)
3: (0.0014052, 0.0027796, 0.0014052, 0.0027796, -0.0010259, 0.0010259)
4: (-0.0045461, -0.0027656, -0.0045461, -0.0027656, -0.0013075, 0.0013075)
5: (-0.0002610, 0.0009141, -0.0002610, 0.0009141, -0.0009636, 0.0009636)
6: (-0.0048016, -0.0016456, -0.0048016, -0.0016456, -0.0022667, 0.0022667)
7: (-0.0222935, -0.0120332, -0.0222935, -0.0120332, -0.0075738, 0.0075738)
8: (0.9751113, 0.9846416, 0.9751113, 0.9846416, -0.0072580, 0.0072580)
9: (0.0001429, 0.0069006, 0.0001429, 0.0069006, -0.0050181, 0.0050181)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.31 + 1.54 = 2.85 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0049498, upper bound: 0.0049498

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048162, upper bound: 0.0048157
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048157, upper bound: 0.0048162
time: 0.69 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.52 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.52
Output dim: 8, lower bound: -0.0048162, upper bound: 0.0048157
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.52
Output dim: 8, lower bound: -0.0048157, upper bound: 0.0048162

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0153504, 0.0180217, 0.0153504, 0.0180217, -0.0021238, 0.0021290
1: -0.0017504, 0.0001848, -0.0017504, 0.0001848, -0.0015699, 0.0015710
2: 0.0036439, 0.0045030, 0.0036439, 0.0045030, -0.0006799, 0.0006779
3: 0.0014052, 0.0027796, 0.0014052, 0.0027796, -0.0009757, 0.0009718
4: -0.0045461, -0.0027656, -0.0045461, -0.0027656, -0.0012348, 0.0012249
5: -0.0002610, 0.0009141, -0.0002610, 0.0009141, -0.0009608, 0.0009604
6: -0.0048016, -0.0016456, -0.0048016, -0.0016456, -0.0021004, 0.0021022
7: -0.0222935, -0.0120332, -0.0222935, -0.0120332, -0.0071645, 0.0071054
8: 0.9751113, 0.9846416, 0.9751113, 0.9846416, -0.0069310, 0.0068813
9: 0.0001429, 0.0069006, 0.0001429, 0.0069006, -0.0047175, 0.0047562

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047464, upper bound: 0.0047463
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047464, upper bound: 0.0047463
time: 0.74 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0153504, 0.0180217, 0.0153504, 0.0180217, -0.0021290, 0.0021238
1: -0.0017504, 0.0001848, -0.0017504, 0.0001848, -0.0015710, 0.0015699
2: 0.0036439, 0.0045030, 0.0036439, 0.0045030, -0.0006779, 0.0006799
3: 0.0014052, 0.0027796, 0.0014052, 0.0027796, -0.0009718, 0.0009757
4: -0.0045461, -0.0027656, -0.0045461, -0.0027656, -0.0012249, 0.0012348
5: -0.0002610, 0.0009141, -0.0002610, 0.0009141, -0.0009604, 0.0009608
6: -0.0048016, -0.0016456, -0.0048016, -0.0016456, -0.0021022, 0.0021004
7: -0.0222935, -0.0120332, -0.0222935, -0.0120332, -0.0071054, 0.0071645
8: 0.9751113, 0.9846416, 0.9751113, 0.9846416, -0.0068813, 0.0069310
9: 0.0001429, 0.0069006, 0.0001429, 0.0069006, -0.0047562, 0.0047175

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047463, upper bound: 0.0047464
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047463, upper bound: 0.0047464
time: 0.70 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.74 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 8, lower bound: -0.0047464, upper bound: 0.0047463
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 8, lower bound: -0.0047464, upper bound: 0.0047463
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 8, lower bound: -0.0047463, upper bound: 0.0047464
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 8, lower bound: -0.0047463, upper bound: 0.0047464

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0153504, 0.0180217, 0.0153504, 0.0180217, -0.0020701, 0.0020999
1: -0.0017504, 0.0001848, -0.0017504, 0.0001848, -0.0015264, 0.0015497
2: 0.0036439, 0.0045030, 0.0036439, 0.0045030, -0.0006702, 0.0006612
3: 0.0014052, 0.0027796, 0.0014052, 0.0027796, -0.0009562, 0.0009428
4: -0.0045461, -0.0027656, -0.0045461, -0.0027656, -0.0012285, 0.0012164
5: -0.0002610, 0.0009141, -0.0002610, 0.0009141, -0.0009474, 0.0009331
6: -0.0048016, -0.0016456, -0.0048016, -0.0016456, -0.0020700, 0.0020706
7: -0.0222935, -0.0120332, -0.0222935, -0.0120332, -0.0071226, 0.0070463
8: 0.9751113, 0.9846416, 0.9751113, 0.9846416, -0.0068637, 0.0067852
9: 0.0001429, 0.0069006, 0.0001429, 0.0069006, -0.0046716, 0.0047237

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045387, upper bound: 0.0045872
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045874, upper bound: 0.0045383
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0153504, 0.0180217, 0.0153504, 0.0180217, -0.0020964, 0.0020753
1: -0.0017504, 0.0001848, -0.0017504, 0.0001848, -0.0015487, 0.0015275
2: 0.0036439, 0.0045030, 0.0036439, 0.0045030, -0.0006632, 0.0006689
3: 0.0014052, 0.0027796, 0.0014052, 0.0027796, -0.0009468, 0.0009518
4: -0.0045461, -0.0027656, -0.0045461, -0.0027656, -0.0012262, 0.0012181
5: -0.0002610, 0.0009141, -0.0002610, 0.0009141, -0.0009334, 0.0009470
6: -0.0048016, -0.0016456, -0.0048016, -0.0016456, -0.0020688, 0.0020710
7: -0.0222935, -0.0120332, -0.0222935, -0.0120332, -0.0071054, 0.0070609
8: 0.9751113, 0.9846416, 0.9751113, 0.9846416, -0.0068349, 0.0068135
9: 0.0001429, 0.0069006, 0.0001429, 0.0069006, -0.0046833, 0.0047103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045387, upper bound: 0.0045872
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045874, upper bound: 0.0045383
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0153504, 0.0180217, 0.0153504, 0.0180217, -0.0020753, 0.0020964
1: -0.0017504, 0.0001848, -0.0017504, 0.0001848, -0.0015275, 0.0015487
2: 0.0036439, 0.0045030, 0.0036439, 0.0045030, -0.0006689, 0.0006632
3: 0.0014052, 0.0027796, 0.0014052, 0.0027796, -0.0009518, 0.0009468
4: -0.0045461, -0.0027656, -0.0045461, -0.0027656, -0.0012181, 0.0012262
5: -0.0002610, 0.0009141, -0.0002610, 0.0009141, -0.0009470, 0.0009334
6: -0.0048016, -0.0016456, -0.0048016, -0.0016456, -0.0020710, 0.0020688
7: -0.0222935, -0.0120332, -0.0222935, -0.0120332, -0.0070609, 0.0071054
8: 0.9751113, 0.9846416, 0.9751113, 0.9846416, -0.0068135, 0.0068349
9: 0.0001429, 0.0069006, 0.0001429, 0.0069006, -0.0047103, 0.0046833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045383, upper bound: 0.0045874
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045872, upper bound: 0.0045387
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0153504, 0.0180217, 0.0153504, 0.0180217, -0.0020999, 0.0020701
1: -0.0017504, 0.0001848, -0.0017504, 0.0001848, -0.0015497, 0.0015264
2: 0.0036439, 0.0045030, 0.0036439, 0.0045030, -0.0006612, 0.0006702
3: 0.0014052, 0.0027796, 0.0014052, 0.0027796, -0.0009428, 0.0009562
4: -0.0045461, -0.0027656, -0.0045461, -0.0027656, -0.0012164, 0.0012285
5: -0.0002610, 0.0009141, -0.0002610, 0.0009141, -0.0009331, 0.0009474
6: -0.0048016, -0.0016456, -0.0048016, -0.0016456, -0.0020706, 0.0020700
7: -0.0222935, -0.0120332, -0.0222935, -0.0120332, -0.0070463, 0.0071226
8: 0.9751113, 0.9846416, 0.9751113, 0.9846416, -0.0067852, 0.0068637
9: 0.0001429, 0.0069006, 0.0001429, 0.0069006, -0.0047237, 0.0046716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045383, upper bound: 0.0045874
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045872, upper bound: 0.0045387
time: 0.71 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.73 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 8, lower bound: -0.0045387, upper bound: 0.0045872
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 8, lower bound: -0.0045874, upper bound: 0.0045383
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 8, lower bound: -0.0045387, upper bound: 0.0045872
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 8, lower bound: -0.0045874, upper bound: 0.0045383
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 8, lower bound: -0.0045383, upper bound: 0.0045874
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 8, lower bound: -0.0045872, upper bound: 0.0045387
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 8, lower bound: -0.0045383, upper bound: 0.0045874
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 8, lower bound: -0.0045872, upper bound: 0.0045387

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0153504, 0.0180217, 0.0153504, 0.0180217, -0.0020387, 0.0020629
1: -0.0017504, 0.0001848, -0.0017504, 0.0001848, -0.0015054, 0.0015271
2: 0.0036439, 0.0045030, 0.0036439, 0.0045030, -0.0006579, 0.0006509
3: 0.0014052, 0.0027796, 0.0014052, 0.0027796, -0.0009506, 0.0009362
4: -0.0045461, -0.0027656, -0.0045461, -0.0027656, -0.0011927, 0.0011850
5: -0.0002610, 0.0009141, -0.0002610, 0.0009141, -0.0009348, 0.0009209
6: -0.0048016, -0.0016456, -0.0048016, -0.0016456, -0.0020271, 0.0020225
7: -0.0222935, -0.0120332, -0.0222935, -0.0120332, -0.0069170, 0.0068684
8: 0.9751113, 0.9846416, 0.9751113, 0.9846416, -0.0066736, 0.0066240
9: 0.0001429, 0.0069006, 0.0001429, 0.0069006, -0.0045558, 0.0045887

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044218, upper bound: 0.0045444
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044950, upper bound: 0.0044560
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0153504, 0.0180217, 0.0153504, 0.0180217, -0.0020331, 0.0020654
1: -0.0017504, 0.0001848, -0.0017504, 0.0001848, -0.0015038, 0.0015272
2: 0.0036439, 0.0045030, 0.0036439, 0.0045030, -0.0006591, 0.0006488
3: 0.0014052, 0.0027796, 0.0014052, 0.0027796, -0.0009496, 0.0009369
4: -0.0045461, -0.0027656, -0.0045461, -0.0027656, -0.0011961, 0.0011806
5: -0.0002610, 0.0009141, -0.0002610, 0.0009141, -0.0009344, 0.0009204
6: -0.0048016, -0.0016456, -0.0048016, -0.0016456, -0.0020218, 0.0020259
7: -0.0222935, -0.0120332, -0.0222935, -0.0120332, -0.0069363, 0.0068408
8: 0.9751113, 0.9846416, 0.9751113, 0.9846416, -0.0066885, 0.0065952
9: 0.0001429, 0.0069006, 0.0001429, 0.0069006, -0.0045367, 0.0046016

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044634, upper bound: 0.0044947
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045444, upper bound: 0.0044134
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0153504, 0.0180217, 0.0153504, 0.0180217, -0.0020628, 0.0020383
1: -0.0017504, 0.0001848, -0.0017504, 0.0001848, -0.0015265, 0.0015050
2: 0.0036439, 0.0045030, 0.0036439, 0.0045030, -0.0006508, 0.0006579
3: 0.0014052, 0.0027796, 0.0014052, 0.0027796, -0.0009410, 0.0009452
4: -0.0045461, -0.0027656, -0.0045461, -0.0027656, -0.0011904, 0.0011866
5: -0.0002610, 0.0009141, -0.0002610, 0.0009141, -0.0009208, 0.0009341
6: -0.0048016, -0.0016456, -0.0048016, -0.0016456, -0.0020266, 0.0020228
7: -0.0222935, -0.0120332, -0.0222935, -0.0120332, -0.0068998, 0.0068819
8: 0.9751113, 0.9846416, 0.9751113, 0.9846416, -0.0066448, 0.0066505
9: 0.0001429, 0.0069006, 0.0001429, 0.0069006, -0.0045673, 0.0045753

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044218, upper bound: 0.0045444
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044950, upper bound: 0.0044564
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0153504, 0.0180217, 0.0153504, 0.0180217, -0.0020594, 0.0020442
1: -0.0017504, 0.0001848, -0.0017504, 0.0001848, -0.0015262, 0.0015065
2: 0.0036439, 0.0045030, 0.0036439, 0.0045030, -0.0006529, 0.0006565
3: 0.0014052, 0.0027796, 0.0014052, 0.0027796, -0.0009402, 0.0009461
4: -0.0045461, -0.0027656, -0.0045461, -0.0027656, -0.0011938, 0.0011823
5: -0.0002610, 0.0009141, -0.0002610, 0.0009141, -0.0009213, 0.0009344
6: -0.0048016, -0.0016456, -0.0048016, -0.0016456, -0.0020207, 0.0020273
7: -0.0222935, -0.0120332, -0.0222935, -0.0120332, -0.0069193, 0.0068553
8: 0.9751113, 0.9846416, 0.9751113, 0.9846416, -0.0066602, 0.0066234
9: 0.0001429, 0.0069006, 0.0001429, 0.0069006, -0.0045483, 0.0045878

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044628, upper bound: 0.0044947
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045444, upper bound: 0.0044134
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0153504, 0.0180217, 0.0153504, 0.0180217, -0.0020442, 0.0020594
1: -0.0017504, 0.0001848, -0.0017504, 0.0001848, -0.0015065, 0.0015262
2: 0.0036439, 0.0045030, 0.0036439, 0.0045030, -0.0006565, 0.0006529
3: 0.0014052, 0.0027796, 0.0014052, 0.0027796, -0.0009461, 0.0009402
4: -0.0045461, -0.0027656, -0.0045461, -0.0027656, -0.0011823, 0.0011938
5: -0.0002610, 0.0009141, -0.0002610, 0.0009141, -0.0009344, 0.0009213
6: -0.0048016, -0.0016456, -0.0048016, -0.0016456, -0.0020273, 0.0020207
7: -0.0222935, -0.0120332, -0.0222935, -0.0120332, -0.0068553, 0.0069193
8: 0.9751113, 0.9846416, 0.9751113, 0.9846416, -0.0066234, 0.0066602
9: 0.0001429, 0.0069006, 0.0001429, 0.0069006, -0.0045878, 0.0045483

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044134, upper bound: 0.0045445
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044947, upper bound: 0.0044628
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0153504, 0.0180217, 0.0153504, 0.0180217, -0.0020383, 0.0020628
1: -0.0017504, 0.0001848, -0.0017504, 0.0001848, -0.0015050, 0.0015265
2: 0.0036439, 0.0045030, 0.0036439, 0.0045030, -0.0006579, 0.0006508
3: 0.0014052, 0.0027796, 0.0014052, 0.0027796, -0.0009452, 0.0009410
4: -0.0045461, -0.0027656, -0.0045461, -0.0027656, -0.0011866, 0.0011904
5: -0.0002610, 0.0009141, -0.0002610, 0.0009141, -0.0009341, 0.0009208
6: -0.0048016, -0.0016456, -0.0048016, -0.0016456, -0.0020228, 0.0020266
7: -0.0222935, -0.0120332, -0.0222935, -0.0120332, -0.0068819, 0.0068998
8: 0.9751113, 0.9846416, 0.9751113, 0.9846416, -0.0066505, 0.0066448
9: 0.0001429, 0.0069006, 0.0001429, 0.0069006, -0.0045753, 0.0045673

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044564, upper bound: 0.0044950
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045444, upper bound: 0.0044218
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0153504, 0.0180217, 0.0153504, 0.0180217, -0.0020654, 0.0020331
1: -0.0017504, 0.0001848, -0.0017504, 0.0001848, -0.0015272, 0.0015038
2: 0.0036439, 0.0045030, 0.0036439, 0.0045030, -0.0006488, 0.0006591
3: 0.0014052, 0.0027796, 0.0014052, 0.0027796, -0.0009369, 0.0009496
4: -0.0045461, -0.0027656, -0.0045461, -0.0027656, -0.0011806, 0.0011961
5: -0.0002610, 0.0009141, -0.0002610, 0.0009141, -0.0009204, 0.0009344
6: -0.0048016, -0.0016456, -0.0048016, -0.0016456, -0.0020259, 0.0020218
7: -0.0222935, -0.0120332, -0.0222935, -0.0120332, -0.0068408, 0.0069363
8: 0.9751113, 0.9846416, 0.9751113, 0.9846416, -0.0065952, 0.0066885
9: 0.0001429, 0.0069006, 0.0001429, 0.0069006, -0.0046016, 0.0045367

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044134, upper bound: 0.0045445
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044947, upper bound: 0.0044634
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0153504, 0.0180217, 0.0153504, 0.0180217, -0.0020629, 0.0020387
1: -0.0017504, 0.0001848, -0.0017504, 0.0001848, -0.0015271, 0.0015054
2: 0.0036439, 0.0045030, 0.0036439, 0.0045030, -0.0006509, 0.0006579
3: 0.0014052, 0.0027796, 0.0014052, 0.0027796, -0.0009362, 0.0009506
4: -0.0045461, -0.0027656, -0.0045461, -0.0027656, -0.0011850, 0.0011927
5: -0.0002610, 0.0009141, -0.0002610, 0.0009141, -0.0009209, 0.0009348
6: -0.0048016, -0.0016456, -0.0048016, -0.0016456, -0.0020225, 0.0020271
7: -0.0222935, -0.0120332, -0.0222935, -0.0120332, -0.0068684, 0.0069171
8: 0.9751113, 0.9846416, 0.9751113, 0.9846416, -0.0066240, 0.0066736
9: 0.0001429, 0.0069006, 0.0001429, 0.0069006, -0.0045887, 0.0045558

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044560, upper bound: 0.0044950
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045444, upper bound: 0.0044218
time: 0.75 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.82 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.82
Output dim: 8, lower bound: -0.0044218, upper bound: 0.0045444
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.82
Output dim: 8, lower bound: -0.0044950, upper bound: 0.0044560
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.82
Output dim: 8, lower bound: -0.0044634, upper bound: 0.0044947
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.82
Output dim: 8, lower bound: -0.0045444, upper bound: 0.0044134
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.82
Output dim: 8, lower bound: -0.0044218, upper bound: 0.0045444
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.82
Output dim: 8, lower bound: -0.0044950, upper bound: 0.0044564
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.82
Output dim: 8, lower bound: -0.0044628, upper bound: 0.0044947
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.82
Output dim: 8, lower bound: -0.0045444, upper bound: 0.0044134
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.82
Output dim: 8, lower bound: -0.0044134, upper bound: 0.0045445
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.82
Output dim: 8, lower bound: -0.0044947, upper bound: 0.0044628
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.82
Output dim: 8, lower bound: -0.0044564, upper bound: 0.0044950
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.82
Output dim: 8, lower bound: -0.0045444, upper bound: 0.0044218
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.82
Output dim: 8, lower bound: -0.0044134, upper bound: 0.0045445
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.82
Output dim: 8, lower bound: -0.0044947, upper bound: 0.0044634
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.82
Output dim: 8, lower bound: -0.0044560, upper bound: 0.0044950
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.82
Output dim: 8, lower bound: -0.0045444, upper bound: 0.0044218

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0153504, 0.0180217, 0.0153504, 0.0180217, -0.0019493, 0.0019663
1: -0.0017504, 0.0001848, -0.0017504, 0.0001848, -0.0014563, 0.0014764
2: 0.0036439, 0.0045030, 0.0036439, 0.0045030, -0.0006240, 0.0006196
3: 0.0014052, 0.0027796, 0.0014052, 0.0027796, -0.0009349, 0.0009179
4: -0.0045461, -0.0027656, -0.0045461, -0.0027656, -0.0010494, 0.0010503
5: -0.0002610, 0.0009141, -0.0002610, 0.0009141, -0.0009087, 0.0008946
6: -0.0048016, -0.0016456, -0.0048016, -0.0016456, -0.0018449, 0.0018280
7: -0.0222935, -0.0120332, -0.0222935, -0.0120332, -0.0060952, 0.0060985
8: 0.9751113, 0.9846416, 0.9751113, 0.9846416, -0.0059702, 0.0059747
9: 0.0001429, 0.0069006, 0.0001429, 0.0069006, -0.0040565, 0.0040527

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032775, upper bound: 0.0033250
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032775, upper bound: 0.0033250
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0153504, 0.0180217, 0.0153504, 0.0180217, -0.0019421, 0.0019708
1: -0.0017504, 0.0001848, -0.0017504, 0.0001848, -0.0014547, 0.0014764
2: 0.0036439, 0.0045030, 0.0036439, 0.0045030, -0.0006261, 0.0006170
3: 0.0014052, 0.0027796, 0.0014052, 0.0027796, -0.0009322, 0.0009200
4: -0.0045461, -0.0027656, -0.0045461, -0.0027656, -0.0010539, 0.0010417
5: -0.0002610, 0.0009141, -0.0002610, 0.0009141, -0.0009080, 0.0008948
6: -0.0048016, -0.0016456, -0.0048016, -0.0016456, -0.0018326, 0.0018372
7: -0.0222935, -0.0120332, -0.0222935, -0.0120332, -0.0061186, 0.0060465
8: 0.9751113, 0.9846416, 0.9751113, 0.9846416, -0.0060030, 0.0059206
9: 0.0001429, 0.0069006, 0.0001429, 0.0069006, -0.0040197, 0.0040706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033193, upper bound: 0.0032780
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033193, upper bound: 0.0032780
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0153504, 0.0180217, 0.0153504, 0.0180217, -0.0019425, 0.0019688
1: -0.0017504, 0.0001848, -0.0017504, 0.0001848, -0.0014534, 0.0014764
2: 0.0036439, 0.0045030, 0.0036439, 0.0045030, -0.0006252, 0.0006171
3: 0.0014052, 0.0027796, 0.0014052, 0.0027796, -0.0009338, 0.0009186
4: -0.0045461, -0.0027656, -0.0045461, -0.0027656, -0.0010529, 0.0010441
5: -0.0002610, 0.0009141, -0.0002610, 0.0009141, -0.0009083, 0.0008937
6: -0.0048016, -0.0016456, -0.0048016, -0.0016456, -0.0018386, 0.0018314
7: -0.0222935, -0.0120332, -0.0222935, -0.0120332, -0.0061144, 0.0060603
8: 0.9751113, 0.9846416, 0.9751113, 0.9846416, -0.0059851, 0.0059383
9: 0.0001429, 0.0069006, 0.0001429, 0.0069006, -0.0040304, 0.0040655

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032775, upper bound: 0.0033250
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032775, upper bound: 0.0033250
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0153504, 0.0180217, 0.0153504, 0.0180217, -0.0019366, 0.0019742
1: -0.0017504, 0.0001848, -0.0017504, 0.0001848, -0.0014531, 0.0014769
2: 0.0036439, 0.0045030, 0.0036439, 0.0045030, -0.0006274, 0.0006150
3: 0.0014052, 0.0027796, 0.0014052, 0.0027796, -0.0009313, 0.0009208
4: -0.0045461, -0.0027656, -0.0045461, -0.0027656, -0.0010594, 0.0010373
5: -0.0002610, 0.0009141, -0.0002610, 0.0009141, -0.0009081, 0.0008944
6: -0.0048016, -0.0016456, -0.0048016, -0.0016456, -0.0018273, 0.0018406
7: -0.0222935, -0.0120332, -0.0222935, -0.0120332, -0.0061518, 0.0060189
8: 0.9751113, 0.9846416, 0.9751113, 0.9846416, -0.0060326, 0.0058918
9: 0.0001429, 0.0069006, 0.0001429, 0.0069006, -0.0040006, 0.0040922

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033193, upper bound: 0.0032780
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033193, upper bound: 0.0032780
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0153504, 0.0180217, 0.0153504, 0.0180217, -0.0019717, 0.0019417
1: -0.0017504, 0.0001848, -0.0017504, 0.0001848, -0.0014761, 0.0014542
2: 0.0036439, 0.0045030, 0.0036439, 0.0045030, -0.0006170, 0.0006263
3: 0.0014052, 0.0027796, 0.0014052, 0.0027796, -0.0009255, 0.0009268
4: -0.0045461, -0.0027656, -0.0045461, -0.0027656, -0.0010472, 0.0010519
5: -0.0002610, 0.0009141, -0.0002610, 0.0009141, -0.0008947, 0.0009078
6: -0.0048016, -0.0016456, -0.0048016, -0.0016456, -0.0018412, 0.0018283
7: -0.0222935, -0.0120332, -0.0222935, -0.0120332, -0.0060779, 0.0061136
8: 0.9751113, 0.9846416, 0.9751113, 0.9846416, -0.0059414, 0.0060041
9: 0.0001429, 0.0069006, 0.0001429, 0.0069006, -0.0040693, 0.0040393

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032775, upper bound: 0.0033250
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032775, upper bound: 0.0033250
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0153504, 0.0180217, 0.0153504, 0.0180217, -0.0019662, 0.0019466
1: -0.0017504, 0.0001848, -0.0017504, 0.0001848, -0.0014757, 0.0014549
2: 0.0036439, 0.0045030, 0.0036439, 0.0045030, -0.0006186, 0.0006241
3: 0.0014052, 0.0027796, 0.0014052, 0.0027796, -0.0009227, 0.0009290
4: -0.0045461, -0.0027656, -0.0045461, -0.0027656, -0.0010522, 0.0010433
5: -0.0002610, 0.0009141, -0.0002610, 0.0009141, -0.0008942, 0.0009080
6: -0.0048016, -0.0016456, -0.0048016, -0.0016456, -0.0018321, 0.0018386
7: -0.0222935, -0.0120332, -0.0222935, -0.0120332, -0.0061065, 0.0060600
8: 0.9751113, 0.9846416, 0.9751113, 0.9846416, -0.0059705, 0.0059471
9: 0.0001429, 0.0069006, 0.0001429, 0.0069006, -0.0040312, 0.0040591

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033193, upper bound: 0.0032780
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033193, upper bound: 0.0032780
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0153504, 0.0180217, 0.0153504, 0.0180217, -0.0019675, 0.0019476
1: -0.0017504, 0.0001848, -0.0017504, 0.0001848, -0.0014753, 0.0014557
2: 0.0036439, 0.0045030, 0.0036439, 0.0045030, -0.0006191, 0.0006249
3: 0.0014052, 0.0027796, 0.0014052, 0.0027796, -0.0009246, 0.0009277
4: -0.0045461, -0.0027656, -0.0045461, -0.0027656, -0.0010506, 0.0010458
5: -0.0002610, 0.0009141, -0.0002610, 0.0009141, -0.0008952, 0.0009077
6: -0.0048016, -0.0016456, -0.0048016, -0.0016456, -0.0018370, 0.0018327
7: -0.0222935, -0.0120332, -0.0222935, -0.0120332, -0.0060974, 0.0060759
8: 0.9751113, 0.9846416, 0.9751113, 0.9846416, -0.0059568, 0.0059674
9: 0.0001429, 0.0069006, 0.0001429, 0.0069006, -0.0040442, 0.0040518

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032775, upper bound: 0.0033250
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032775, upper bound: 0.0033250
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0153504, 0.0180217, 0.0153504, 0.0180217, -0.0019628, 0.0019523
1: -0.0017504, 0.0001848, -0.0017504, 0.0001848, -0.0014754, 0.0014575
2: 0.0036439, 0.0045030, 0.0036439, 0.0045030, -0.0006207, 0.0006227
3: 0.0014052, 0.0027796, 0.0014052, 0.0027796, -0.0009218, 0.0009299
4: -0.0045461, -0.0027656, -0.0045461, -0.0027656, -0.0010578, 0.0010391
5: -0.0002610, 0.0009141, -0.0002610, 0.0009141, -0.0008951, 0.0009083
6: -0.0048016, -0.0016456, -0.0048016, -0.0016456, -0.0018262, 0.0018432
7: -0.0222935, -0.0120332, -0.0222935, -0.0120332, -0.0061384, 0.0060334
8: 0.9751113, 0.9846416, 0.9751113, 0.9846416, -0.0060016, 0.0059200
9: 0.0001429, 0.0069006, 0.0001429, 0.0069006, -0.0040123, 0.0040804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033193, upper bound: 0.0032780
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033193, upper bound: 0.0032780
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0153504, 0.0180217, 0.0153504, 0.0180217, -0.0019523, 0.0019628
1: -0.0017504, 0.0001848, -0.0017504, 0.0001848, -0.0014575, 0.0014754
2: 0.0036439, 0.0045030, 0.0036439, 0.0045030, -0.0006227, 0.0006207
3: 0.0014052, 0.0027796, 0.0014052, 0.0027796, -0.0009299, 0.0009218
4: -0.0045461, -0.0027656, -0.0045461, -0.0027656, -0.0010391, 0.0010578
5: -0.0002610, 0.0009141, -0.0002610, 0.0009141, -0.0009083, 0.0008951
6: -0.0048016, -0.0016456, -0.0048016, -0.0016456, -0.0018432, 0.0018262
7: -0.0222935, -0.0120332, -0.0222935, -0.0120332, -0.0060334, 0.0061384
8: 0.9751113, 0.9846416, 0.9751113, 0.9846416, -0.0059200, 0.0060016
9: 0.0001429, 0.0069006, 0.0001429, 0.0069006, -0.0040804, 0.0040123

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032780, upper bound: 0.0033193
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032780, upper bound: 0.0033193
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0153504, 0.0180217, 0.0153504, 0.0180217, -0.0019476, 0.0019675
1: -0.0017504, 0.0001848, -0.0017504, 0.0001848, -0.0014557, 0.0014753
2: 0.0036439, 0.0045030, 0.0036439, 0.0045030, -0.0006249, 0.0006191
3: 0.0014052, 0.0027796, 0.0014052, 0.0027796, -0.0009277, 0.0009246
4: -0.0045461, -0.0027656, -0.0045461, -0.0027656, -0.0010458, 0.0010506
5: -0.0002610, 0.0009141, -0.0002610, 0.0009141, -0.0009077, 0.0008952
6: -0.0048016, -0.0016456, -0.0048016, -0.0016456, -0.0018327, 0.0018370
7: -0.0222935, -0.0120332, -0.0222935, -0.0120332, -0.0060759, 0.0060974
8: 0.9751113, 0.9846416, 0.9751113, 0.9846416, -0.0059674, 0.0059568
9: 0.0001429, 0.0069006, 0.0001429, 0.0069006, -0.0040518, 0.0040442

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033250, upper bound: 0.0032775
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033250, upper bound: 0.0032775
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0153504, 0.0180217, 0.0153504, 0.0180217, -0.0019466, 0.0019662
1: -0.0017504, 0.0001848, -0.0017504, 0.0001848, -0.0014549, 0.0014757
2: 0.0036439, 0.0045030, 0.0036439, 0.0045030, -0.0006241, 0.0006186
3: 0.0014052, 0.0027796, 0.0014052, 0.0027796, -0.0009290, 0.0009227
4: -0.0045461, -0.0027656, -0.0045461, -0.0027656, -0.0010433, 0.0010522
5: -0.0002610, 0.0009141, -0.0002610, 0.0009141, -0.0009080, 0.0008942
6: -0.0048016, -0.0016456, -0.0048016, -0.0016456, -0.0018386, 0.0018321
7: -0.0222935, -0.0120332, -0.0222935, -0.0120332, -0.0060600, 0.0061065
8: 0.9751113, 0.9846416, 0.9751113, 0.9846416, -0.0059472, 0.0059705
9: 0.0001429, 0.0069006, 0.0001429, 0.0069006, -0.0040591, 0.0040313

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032780, upper bound: 0.0033193
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032780, upper bound: 0.0033193
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0153504, 0.0180217, 0.0153504, 0.0180217, -0.0019417, 0.0019717
1: -0.0017504, 0.0001848, -0.0017504, 0.0001848, -0.0014542, 0.0014761
2: 0.0036439, 0.0045030, 0.0036439, 0.0045030, -0.0006263, 0.0006170
3: 0.0014052, 0.0027796, 0.0014052, 0.0027796, -0.0009268, 0.0009255
4: -0.0045461, -0.0027656, -0.0045461, -0.0027656, -0.0010519, 0.0010472
5: -0.0002610, 0.0009141, -0.0002610, 0.0009141, -0.0009078, 0.0008947
6: -0.0048016, -0.0016456, -0.0048016, -0.0016456, -0.0018283, 0.0018412
7: -0.0222935, -0.0120332, -0.0222935, -0.0120332, -0.0061136, 0.0060779
8: 0.9751113, 0.9846416, 0.9751113, 0.9846416, -0.0060041, 0.0059414
9: 0.0001429, 0.0069006, 0.0001429, 0.0069006, -0.0040393, 0.0040693

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033250, upper bound: 0.0032775
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033250, upper bound: 0.0032775
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0153504, 0.0180217, 0.0153504, 0.0180217, -0.0019742, 0.0019366
1: -0.0017504, 0.0001848, -0.0017504, 0.0001848, -0.0014769, 0.0014531
2: 0.0036439, 0.0045030, 0.0036439, 0.0045030, -0.0006150, 0.0006274
3: 0.0014052, 0.0027796, 0.0014052, 0.0027796, -0.0009208, 0.0009313
4: -0.0045461, -0.0027656, -0.0045461, -0.0027656, -0.0010373, 0.0010594
5: -0.0002610, 0.0009141, -0.0002610, 0.0009141, -0.0008944, 0.0009081
6: -0.0048016, -0.0016456, -0.0048016, -0.0016456, -0.0018406, 0.0018273
7: -0.0222935, -0.0120332, -0.0222935, -0.0120332, -0.0060189, 0.0061518
8: 0.9751113, 0.9846416, 0.9751113, 0.9846416, -0.0058918, 0.0060326
9: 0.0001429, 0.0069006, 0.0001429, 0.0069006, -0.0040922, 0.0040006

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032780, upper bound: 0.0033193
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032780, upper bound: 0.0033193
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0153504, 0.0180217, 0.0153504, 0.0180217, -0.0019688, 0.0019425
1: -0.0017504, 0.0001848, -0.0017504, 0.0001848, -0.0014764, 0.0014534
2: 0.0036439, 0.0045030, 0.0036439, 0.0045030, -0.0006171, 0.0006252
3: 0.0014052, 0.0027796, 0.0014052, 0.0027796, -0.0009186, 0.0009338
4: -0.0045461, -0.0027656, -0.0045461, -0.0027656, -0.0010441, 0.0010529
5: -0.0002610, 0.0009141, -0.0002610, 0.0009141, -0.0008937, 0.0009083
6: -0.0048016, -0.0016456, -0.0048016, -0.0016456, -0.0018314, 0.0018386
7: -0.0222935, -0.0120332, -0.0222935, -0.0120332, -0.0060603, 0.0061144
8: 0.9751113, 0.9846416, 0.9751113, 0.9846416, -0.0059383, 0.0059851
9: 0.0001429, 0.0069006, 0.0001429, 0.0069006, -0.0040655, 0.0040304

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033250, upper bound: 0.0032775
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033250, upper bound: 0.0032775
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0153504, 0.0180217, 0.0153504, 0.0180217, -0.0019708, 0.0019421
1: -0.0017504, 0.0001848, -0.0017504, 0.0001848, -0.0014764, 0.0014547
2: 0.0036439, 0.0045030, 0.0036439, 0.0045030, -0.0006170, 0.0006261
3: 0.0014052, 0.0027796, 0.0014052, 0.0027796, -0.0009200, 0.0009322
4: -0.0045461, -0.0027656, -0.0045461, -0.0027656, -0.0010417, 0.0010539
5: -0.0002610, 0.0009141, -0.0002610, 0.0009141, -0.0008948, 0.0009080
6: -0.0048016, -0.0016456, -0.0048016, -0.0016456, -0.0018372, 0.0018326
7: -0.0222935, -0.0120332, -0.0222935, -0.0120332, -0.0060465, 0.0061186
8: 0.9751113, 0.9846416, 0.9751113, 0.9846416, -0.0059206, 0.0060030
9: 0.0001429, 0.0069006, 0.0001429, 0.0069006, -0.0040706, 0.0040197

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032780, upper bound: 0.0033193
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032780, upper bound: 0.0033193
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0153504, 0.0180217, 0.0153504, 0.0180217, -0.0019663, 0.0019493
1: -0.0017504, 0.0001848, -0.0017504, 0.0001848, -0.0014764, 0.0014563
2: 0.0036439, 0.0045030, 0.0036439, 0.0045030, -0.0006196, 0.0006240
3: 0.0014052, 0.0027796, 0.0014052, 0.0027796, -0.0009179, 0.0009349
4: -0.0045461, -0.0027656, -0.0045461, -0.0027656, -0.0010503, 0.0010494
5: -0.0002610, 0.0009141, -0.0002610, 0.0009141, -0.0008946, 0.0009087
6: -0.0048016, -0.0016456, -0.0048016, -0.0016456, -0.0018280, 0.0018449
7: -0.0222935, -0.0120332, -0.0222935, -0.0120332, -0.0060985, 0.0060952
8: 0.9751113, 0.9846416, 0.9751113, 0.9846416, -0.0059747, 0.0059702
9: 0.0001429, 0.0069006, 0.0001429, 0.0069006, -0.0040527, 0.0040565

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033250, upper bound: 0.0032775
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033250, upper bound: 0.0032775
time: 0.60 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.51 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 8, lower bound: -0.0032775, upper bound: 0.0033250
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 8, lower bound: -0.0032775, upper bound: 0.0033250
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 8, lower bound: -0.0033193, upper bound: 0.0032780
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 8, lower bound: -0.0033193, upper bound: 0.0032780
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 8, lower bound: -0.0032775, upper bound: 0.0033250
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 8, lower bound: -0.0032775, upper bound: 0.0033250
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 8, lower bound: -0.0033193, upper bound: 0.0032780
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 8, lower bound: -0.0033193, upper bound: 0.0032780
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 8, lower bound: -0.0032775, upper bound: 0.0033250
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 8, lower bound: -0.0032775, upper bound: 0.0033250
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 8, lower bound: -0.0033193, upper bound: 0.0032780
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 8, lower bound: -0.0033193, upper bound: 0.0032780
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 8, lower bound: -0.0032775, upper bound: 0.0033250
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 8, lower bound: -0.0032775, upper bound: 0.0033250
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 8, lower bound: -0.0033193, upper bound: 0.0032780
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 8, lower bound: -0.0033193, upper bound: 0.0032780
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 8, lower bound: -0.0032780, upper bound: 0.0033193
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 8, lower bound: -0.0032780, upper bound: 0.0033193
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 8, lower bound: -0.0033250, upper bound: 0.0032775
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 8, lower bound: -0.0033250, upper bound: 0.0032775
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 8, lower bound: -0.0032780, upper bound: 0.0033193
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 8, lower bound: -0.0032780, upper bound: 0.0033193
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 8, lower bound: -0.0033250, upper bound: 0.0032775
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 8, lower bound: -0.0033250, upper bound: 0.0032775
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 8, lower bound: -0.0032780, upper bound: 0.0033193
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 8, lower bound: -0.0032780, upper bound: 0.0033193
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 8, lower bound: -0.0033250, upper bound: 0.0032775
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 8, lower bound: -0.0033250, upper bound: 0.0032775
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 8, lower bound: -0.0032780, upper bound: 0.0033193
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 8, lower bound: -0.0032780, upper bound: 0.0033193
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 8, lower bound: -0.0033250, upper bound: 0.0032775
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 8, lower bound: -0.0033250, upper bound: 0.0032775

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.85 + 80.70 = 83.55 seconds
