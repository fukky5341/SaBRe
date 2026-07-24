## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00886005


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0005231, 0.0010227, -0.0005231, 0.0010227, -0.0011624, 0.0011624)
1: (-0.0011286, 0.0028889, -0.0011286, 0.0028889, -0.0029943, 0.0029943)
2: (0.0120135, 0.0180301, 0.0120135, 0.0180301, -0.0041898, 0.0041898)
3: (-0.0015932, 0.0029310, -0.0015932, 0.0029310, -0.0030234, 0.0030234)
4: (-0.0058492, -0.0016761, -0.0058492, -0.0016761, -0.0038423, 0.0038423)
5: (0.0063478, 0.0108638, 0.0063478, 0.0108638, -0.0030064, 0.0030064)
6: (0.0074971, 0.0105374, 0.0074971, 0.0105374, -0.0030403, 0.0030403)
7: (-0.0219836, -0.0121800, -0.0219836, -0.0121800, -0.0056026, 0.0056026)
8: (0.9608052, 0.9888938, 0.9608052, 0.9888938, -0.0198586, 0.0198586)
9: (0.0009172, 0.0091725, 0.0009172, 0.0091725, -0.0049195, 0.0049195)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.27 + 1.47 = 2.74 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0129357, upper bound: 0.0129357

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118007, upper bound: 0.0126550
time: 0.69 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126550, upper bound: 0.0126550
time: 0.71 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.55 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.55
Output dim: 8, lower bound: -0.0118007, upper bound: 0.0126550
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.55
Output dim: 8, lower bound: -0.0126550, upper bound: 0.0126550

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0004717, 0.0010175, -0.0005157, 0.0010219, -0.0011021, 0.0011483
1: -0.0008880, 0.0028810, -0.0010938, 0.0028878, -0.0028536, 0.0029629
2: 0.0120254, 0.0176699, 0.0120152, 0.0179781, -0.0041397, 0.0039578
3: -0.0015843, 0.0026601, -0.0015920, 0.0028919, -0.0029825, 0.0028346
4: -0.0058410, -0.0019260, -0.0058481, -0.0017122, -0.0038144, 0.0037111
5: 0.0063567, 0.0105934, 0.0063491, 0.0108248, -0.0029650, 0.0028160
6: 0.0075090, 0.0105340, 0.0074988, 0.0105369, -0.0030279, 0.0030353
7: -0.0213966, -0.0121992, -0.0218988, -0.0121827, -0.0049435, 0.0054837
8: 0.9624872, 0.9888387, 0.9610481, 0.9888859, -0.0188031, 0.0196307
9: 0.0009335, 0.0086782, 0.0009195, 0.0091011, -0.0048269, 0.0044216

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118007, upper bound: 0.0118007
time: 0.73 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118007, upper bound: 0.0126550
time: 0.76 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0004976, 0.0012486, -0.0005144, 0.0010200, -0.0011284, 0.0014031
1: -0.0010090, 0.0032352, -0.0010877, 0.0028848, -0.0032120, 0.0033959
2: 0.0114949, 0.0178511, 0.0120197, 0.0179690, -0.0048092, 0.0044196
3: -0.0019832, 0.0027964, -0.0015886, 0.0028850, -0.0034955, 0.0031447
4: -0.0062090, -0.0018003, -0.0058450, -0.0017185, -0.0042432, 0.0040447
5: 0.0059585, 0.0107294, 0.0063524, 0.0108179, -0.0034778, 0.0031219
6: 0.0069718, 0.0106843, 0.0075032, 0.0105357, -0.0035639, 0.0031811
7: -0.0216918, -0.0113349, -0.0218840, -0.0121900, -0.0052651, 0.0066838
8: 0.9616413, 0.9913152, 0.9610907, 0.9888653, -0.0210373, 0.0227347
9: 0.0002056, 0.0089268, 0.0009256, 0.0090886, -0.0058104, 0.0047561

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126550, upper bound: 0.0118007
time: 0.74 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126550, upper bound: 0.0126550
time: 0.74 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.98 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.98
Output dim: 8, lower bound: -0.0118007, upper bound: 0.0118007
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.98
Output dim: 8, lower bound: -0.0118007, upper bound: 0.0126550
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.98
Output dim: 8, lower bound: -0.0126550, upper bound: 0.0118007
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.98
Output dim: 8, lower bound: -0.0126550, upper bound: 0.0126550

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004717, 0.0010175, -0.0004717, 0.0010175, -0.0010977, 0.0010977
1: -0.0008880, 0.0028810, -0.0008880, 0.0028810, -0.0028470, 0.0028470
2: 0.0120254, 0.0176699, 0.0120254, 0.0176699, -0.0039479, 0.0039479
3: -0.0015843, 0.0026601, -0.0015843, 0.0026601, -0.0028272, 0.0028272
4: -0.0058410, -0.0019260, -0.0058410, -0.0019260, -0.0037043, 0.0037043
5: 0.0063567, 0.0105934, 0.0063567, 0.0105934, -0.0028087, 0.0028087
6: 0.0075090, 0.0105340, 0.0075090, 0.0105340, -0.0030250, 0.0030250
7: -0.0213966, -0.0121992, -0.0213966, -0.0121992, -0.0049276, 0.0049276
8: 0.9624872, 0.9888387, 0.9624872, 0.9888387, -0.0187573, 0.0187573
9: 0.0009335, 0.0086782, 0.0009335, 0.0086782, -0.0044081, 0.0044081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115608, upper bound: 0.0117932
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115608, upper bound: 0.0118014
time: 0.75 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004717, 0.0010175, -0.0004976, 0.0012486, -0.0013470, 0.0011489
1: -0.0008880, 0.0028810, -0.0010090, 0.0032352, -0.0033123, 0.0030082
2: 0.0120254, 0.0176699, 0.0114949, 0.0178511, -0.0041854, 0.0046448
3: -0.0015843, 0.0026601, -0.0019832, 0.0027964, -0.0030121, 0.0033512
4: -0.0058410, -0.0019260, -0.0062090, -0.0018003, -0.0039703, 0.0041876
5: 0.0063567, 0.0105934, 0.0059585, 0.0107294, -0.0029940, 0.0033317
6: 0.0075090, 0.0105340, 0.0069718, 0.0106843, -0.0031753, 0.0035622
7: -0.0213966, -0.0121992, -0.0216918, -0.0113349, -0.0060630, 0.0055162
8: 0.9624872, 0.9888387, 0.9616413, 0.9913152, -0.0220104, 0.0198526
9: 0.0009335, 0.0086782, 0.0002056, 0.0089268, -0.0048501, 0.0053643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115410, upper bound: 0.0124099
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115608, upper bound: 0.0124099
time: 0.72 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004976, 0.0012486, -0.0004717, 0.0010175, -0.0011489, 0.0013470
1: -0.0010090, 0.0032352, -0.0008880, 0.0028810, -0.0030082, 0.0033123
2: 0.0114949, 0.0178511, 0.0120254, 0.0176699, -0.0046448, 0.0041854
3: -0.0019832, 0.0027964, -0.0015843, 0.0026601, -0.0033512, 0.0030121
4: -0.0062090, -0.0018003, -0.0058410, -0.0019260, -0.0041876, 0.0039703
5: 0.0059585, 0.0107294, 0.0063567, 0.0105934, -0.0033317, 0.0029940
6: 0.0069718, 0.0106843, 0.0075090, 0.0105340, -0.0035622, 0.0031753
7: -0.0216918, -0.0113349, -0.0213966, -0.0121992, -0.0055162, 0.0060630
8: 0.9616413, 0.9913152, 0.9624872, 0.9888387, -0.0198526, 0.0220104
9: 0.0002056, 0.0089268, 0.0009335, 0.0086782, -0.0053643, 0.0048501

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124099, upper bound: 0.0115410
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124099, upper bound: 0.0115608
time: 0.60 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004976, 0.0012486, -0.0004976, 0.0012486, -0.0013071, 0.0013071
1: -0.0010090, 0.0032352, -0.0010090, 0.0032352, -0.0033191, 0.0033191
2: 0.0114949, 0.0178511, 0.0114949, 0.0178511, -0.0045800, 0.0045800
3: -0.0019832, 0.0027964, -0.0019832, 0.0027964, -0.0032653, 0.0032653
4: -0.0062090, -0.0018003, -0.0062090, -0.0018003, -0.0043863, 0.0043863
5: 0.0059585, 0.0107294, 0.0059585, 0.0107294, -0.0032423, 0.0032423
6: 0.0069718, 0.0106843, 0.0069718, 0.0106843, -0.0037125, 0.0037125
7: -0.0216918, -0.0113349, -0.0216918, -0.0113349, -0.0055263, 0.0055263
8: 0.9616413, 0.9913152, 0.9616413, 0.9913152, -0.0217858, 0.0217858
9: 0.0002056, 0.0089268, 0.0002056, 0.0089268, -0.0049761, 0.0049761

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124099, upper bound: 0.0115497
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124099, upper bound: 0.0115690
time: 0.75 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.91 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 8, lower bound: -0.0115608, upper bound: 0.0117932
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 8, lower bound: -0.0115608, upper bound: 0.0118014
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 8, lower bound: -0.0115410, upper bound: 0.0124099
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 8, lower bound: -0.0115608, upper bound: 0.0124099
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 8, lower bound: -0.0124099, upper bound: 0.0115410
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 8, lower bound: -0.0124099, upper bound: 0.0115608
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 8, lower bound: -0.0124099, upper bound: 0.0115497
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 8, lower bound: -0.0124099, upper bound: 0.0115690

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004710, 0.0009799, -0.0004716, 0.0010146, -0.0010506, 0.0010183
1: -0.0008847, 0.0028234, -0.0008877, 0.0028766, -0.0028789, 0.0028848
2: 0.0121117, 0.0176648, 0.0120320, 0.0176694, -0.0039868, 0.0039904
3: -0.0015194, 0.0026563, -0.0015794, 0.0026597, -0.0028457, 0.0028544
4: -0.0057811, -0.0019294, -0.0058364, -0.0019263, -0.0037684, 0.0037351
5: 0.0064215, 0.0105897, 0.0063617, 0.0105931, -0.0028260, 0.0028352
6: 0.0075965, 0.0105096, 0.0075157, 0.0105322, -0.0029357, 0.0029939
7: -0.0213884, -0.0123399, -0.0213958, -0.0122100, -0.0045725, 0.0044763
8: 0.9625106, 0.9884356, 0.9624893, 0.9888078, -0.0189624, 0.0189577
9: 0.0010519, 0.0086713, 0.0009425, 0.0086776, -0.0042614, 0.0043155

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114065, upper bound: 0.0114012
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113876, upper bound: 0.0113947
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004812, 0.0009447, -0.0004714, 0.0009943, -0.0010511, 0.0009924
1: -0.0009324, 0.0027695, -0.0008864, 0.0028454, -0.0028563, 0.0028460
2: 0.0121924, 0.0177363, 0.0120787, 0.0176675, -0.0039365, 0.0039753
3: -0.0014587, 0.0027100, -0.0015442, 0.0026583, -0.0028116, 0.0028504
4: -0.0057251, -0.0018799, -0.0058040, -0.0019276, -0.0037126, 0.0036778
5: 0.0064821, 0.0106433, 0.0063968, 0.0105916, -0.0027923, 0.0028319
6: 0.0076782, 0.0104867, 0.0075630, 0.0105189, -0.0028407, 0.0029237
7: -0.0215048, -0.0124715, -0.0213927, -0.0122862, -0.0047274, 0.0044436
8: 0.9621772, 0.9880587, 0.9624981, 0.9885898, -0.0188727, 0.0187147
9: 0.0011627, 0.0087693, 0.0010066, 0.0086750, -0.0042184, 0.0043846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114065, upper bound: 0.0113958
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113842, upper bound: 0.0113842
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0004716, 0.0010146, -0.0004965, 0.0012114, -0.0012671, 0.0011441
1: -0.0008877, 0.0028766, -0.0010041, 0.0031782, -0.0033736, 0.0029425
2: 0.0120320, 0.0176694, 0.0115803, 0.0178437, -0.0041047, 0.0047188
3: -0.0015794, 0.0026597, -0.0019190, 0.0027908, -0.0029584, 0.0033962
4: -0.0058364, -0.0019263, -0.0061497, -0.0018054, -0.0038685, 0.0042234
5: 0.0063617, 0.0105931, 0.0060227, 0.0107239, -0.0029412, 0.0033755
6: 0.0075157, 0.0105322, 0.0070583, 0.0106601, -0.0031444, 0.0034738
7: -0.0213958, -0.0122100, -0.0216798, -0.0114741, -0.0056691, 0.0054798
8: 0.9624893, 0.9888078, 0.9616759, 0.9909164, -0.0223754, 0.0194604
9: 0.0009425, 0.0086776, 0.0003228, 0.0089167, -0.0048025, 0.0052658

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111856, upper bound: 0.0120282
time: 0.73 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111480, upper bound: 0.0120051
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0004714, 0.0009943, -0.0005078, 0.0011786, -0.0012443, 0.0011334
1: -0.0008864, 0.0028454, -0.0010568, 0.0031279, -0.0033344, 0.0029121
2: 0.0120787, 0.0176675, 0.0116556, 0.0179226, -0.0040737, 0.0046679
3: -0.0015442, 0.0026583, -0.0018624, 0.0028502, -0.0029418, 0.0033616
4: -0.0058040, -0.0019276, -0.0060975, -0.0017506, -0.0037981, 0.0041699
5: 0.0063968, 0.0105916, 0.0060792, 0.0107831, -0.0029254, 0.0033413
6: 0.0075630, 0.0105189, 0.0071345, 0.0106388, -0.0030758, 0.0033844
7: -0.0213927, -0.0122862, -0.0218084, -0.0115967, -0.0056354, 0.0055089
8: 0.9624981, 0.9885898, 0.9613071, 0.9905650, -0.0221292, 0.0193019
9: 0.0010066, 0.0086750, 0.0004261, 0.0090250, -0.0048165, 0.0052219

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111520, upper bound: 0.0119932
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111511, upper bound: 0.0120002
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004965, 0.0012114, -0.0004716, 0.0010146, -0.0011441, 0.0012671
1: -0.0010041, 0.0031782, -0.0008877, 0.0028766, -0.0029425, 0.0033736
2: 0.0115803, 0.0178437, 0.0120320, 0.0176694, -0.0047188, 0.0041047
3: -0.0019190, 0.0027908, -0.0015794, 0.0026597, -0.0033962, 0.0029584
4: -0.0061497, -0.0018054, -0.0058364, -0.0019263, -0.0042234, 0.0038685
5: 0.0060227, 0.0107239, 0.0063617, 0.0105931, -0.0033755, 0.0029412
6: 0.0070583, 0.0106601, 0.0075157, 0.0105322, -0.0034738, 0.0031444
7: -0.0216798, -0.0114741, -0.0213958, -0.0122100, -0.0054797, 0.0056691
8: 0.9616759, 0.9909164, 0.9624893, 0.9888078, -0.0194604, 0.0223754
9: 0.0003228, 0.0089167, 0.0009425, 0.0086776, -0.0052658, 0.0048025

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120282, upper bound: 0.0111856
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120051, upper bound: 0.0111480
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0005078, 0.0011786, -0.0004714, 0.0009943, -0.0011334, 0.0012443
1: -0.0010568, 0.0031279, -0.0008864, 0.0028454, -0.0029121, 0.0033344
2: 0.0116556, 0.0179226, 0.0120787, 0.0176675, -0.0046679, 0.0040737
3: -0.0018624, 0.0028502, -0.0015442, 0.0026583, -0.0033616, 0.0029418
4: -0.0060975, -0.0017506, -0.0058040, -0.0019276, -0.0041699, 0.0037981
5: 0.0060792, 0.0107831, 0.0063968, 0.0105916, -0.0033413, 0.0029254
6: 0.0071345, 0.0106388, 0.0075630, 0.0105189, -0.0033844, 0.0030758
7: -0.0218084, -0.0115967, -0.0213927, -0.0122862, -0.0055089, 0.0056354
8: 0.9613071, 0.9905650, 0.9624981, 0.9885898, -0.0193019, 0.0221292
9: 0.0004261, 0.0090250, 0.0010066, 0.0086750, -0.0052219, 0.0048165

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119932, upper bound: 0.0111520
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120002, upper bound: 0.0111511
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004965, 0.0012114, -0.0004975, 0.0012458, -0.0013026, 0.0012689
1: -0.0010041, 0.0031782, -0.0010086, 0.0032309, -0.0032486, 0.0032562
2: 0.0115803, 0.0178437, 0.0115014, 0.0178505, -0.0044870, 0.0044885
3: -0.0019190, 0.0027908, -0.0019783, 0.0027959, -0.0031958, 0.0032025
4: -0.0061497, -0.0018054, -0.0062044, -0.0018007, -0.0043186, 0.0042814
5: 0.0060227, 0.0107239, 0.0059634, 0.0107290, -0.0031730, 0.0031801
6: 0.0070583, 0.0106601, 0.0069784, 0.0106825, -0.0036241, 0.0036817
7: -0.0216798, -0.0114741, -0.0216909, -0.0113455, -0.0054837, 0.0053831
8: 0.9616759, 0.9909164, 0.9616439, 0.9912848, -0.0213457, 0.0213507
9: 0.0003228, 0.0089167, 0.0002145, 0.0089260, -0.0048531, 0.0049129

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119934, upper bound: 0.0111585
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120051, upper bound: 0.0111585
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0005078, 0.0011786, -0.0004970, 0.0012259, -0.0012974, 0.0012445
1: -0.0010568, 0.0031279, -0.0010066, 0.0032003, -0.0032136, 0.0032064
2: 0.0116556, 0.0179226, 0.0115472, 0.0178475, -0.0044195, 0.0044541
3: -0.0018624, 0.0028502, -0.0019439, 0.0027937, -0.0031484, 0.0031855
4: -0.0060975, -0.0017506, -0.0061727, -0.0018027, -0.0042499, 0.0042050
5: 0.0060792, 0.0107831, 0.0059978, 0.0107267, -0.0031260, 0.0031641
6: 0.0071345, 0.0106388, 0.0070247, 0.0106695, -0.0035349, 0.0036140
7: -0.0218084, -0.0115967, -0.0216860, -0.0114200, -0.0055719, 0.0053332
8: 0.9613071, 0.9905650, 0.9616579, 0.9910712, -0.0211661, 0.0210282
9: 0.0004261, 0.0090250, 0.0002773, 0.0089219, -0.0047962, 0.0049709

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119932, upper bound: 0.0111651
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120002, upper bound: 0.0111647
time: 0.74 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.76 seconds
IS_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 8, lower bound: -0.0114065, upper bound: 0.0114012
IS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 8, lower bound: -0.0113876, upper bound: 0.0113947
IS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 8, lower bound: -0.0114065, upper bound: 0.0113958
IS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 8, lower bound: -0.0113842, upper bound: 0.0113842
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 8, lower bound: -0.0111856, upper bound: 0.0120282
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 8, lower bound: -0.0111480, upper bound: 0.0120051
IS_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 8, lower bound: -0.0111520, upper bound: 0.0119932
IS_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 8, lower bound: -0.0111511, upper bound: 0.0120002
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 8, lower bound: -0.0120282, upper bound: 0.0111856
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 8, lower bound: -0.0120051, upper bound: 0.0111480
IS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 8, lower bound: -0.0119932, upper bound: 0.0111520
IS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 8, lower bound: -0.0120002, upper bound: 0.0111511
IS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 8, lower bound: -0.0119934, upper bound: 0.0111585
IS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 8, lower bound: -0.0120051, upper bound: 0.0111585
IS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 8, lower bound: -0.0119932, upper bound: 0.0111651
IS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 8, lower bound: -0.0120002, upper bound: 0.0111647

## BFS IS instance: IS_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0004619, 0.0009759, -0.0004716, 0.0010146, -0.0010389, 0.0010143
1: -0.0008421, 0.0028172, -0.0008877, 0.0028766, -0.0028360, 0.0028795
2: 0.0121210, 0.0176011, 0.0120320, 0.0176694, -0.0039789, 0.0039266
3: -0.0015124, 0.0026084, -0.0015794, 0.0026597, -0.0028398, 0.0028068
4: -0.0057747, -0.0019736, -0.0058364, -0.0019263, -0.0037630, 0.0036874
5: 0.0064285, 0.0105418, 0.0063617, 0.0105931, -0.0028200, 0.0027877
6: 0.0076059, 0.0105070, 0.0075157, 0.0105322, -0.0029263, 0.0029912
7: -0.0212845, -0.0123551, -0.0213958, -0.0122100, -0.0044479, 0.0044634
8: 0.9628081, 0.9883922, 0.9624893, 0.9888078, -0.0186633, 0.0189208
9: 0.0010647, 0.0085839, 0.0009425, 0.0086776, -0.0042505, 0.0042249

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113869, upper bound: 0.0113935
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113869, upper bound: 0.0113947
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0004366, 0.0010694, -0.0004620, 0.0010102, -0.0010352, 0.0011110
1: -0.0007235, 0.0029606, -0.0008423, 0.0028698, -0.0029761, 0.0031118
2: 0.0119062, 0.0174235, 0.0120421, 0.0176015, -0.0043305, 0.0041034
3: -0.0016739, 0.0024749, -0.0015717, 0.0026087, -0.0031064, 0.0029225
4: -0.0059237, -0.0020968, -0.0058294, -0.0019734, -0.0039503, 0.0037326
5: 0.0062673, 0.0104085, 0.0063693, 0.0105421, -0.0030864, 0.0029017
6: 0.0073884, 0.0105678, 0.0075260, 0.0105293, -0.0031409, 0.0030418
7: -0.0209952, -0.0120051, -0.0212852, -0.0122266, -0.0044931, 0.0050730
8: 0.9636371, 0.9893949, 0.9628063, 0.9887604, -0.0195247, 0.0205589
9: 0.0007700, 0.0083402, 0.0009565, 0.0085844, -0.0047558, 0.0043431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113876, upper bound: 0.0113947
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113876, upper bound: 0.0113947
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0004722, 0.0009407, -0.0004714, 0.0009943, -0.0010407, 0.0009885
1: -0.0008903, 0.0027633, -0.0008864, 0.0028454, -0.0028134, 0.0028409
2: 0.0122016, 0.0176734, 0.0120787, 0.0176675, -0.0039288, 0.0039107
3: -0.0014518, 0.0026627, -0.0015442, 0.0026583, -0.0028058, 0.0028024
4: -0.0057188, -0.0019235, -0.0058040, -0.0019276, -0.0037073, 0.0036303
5: 0.0064890, 0.0105960, 0.0063968, 0.0105916, -0.0027865, 0.0027841
6: 0.0076875, 0.0104841, 0.0075630, 0.0105189, -0.0028314, 0.0029211
7: -0.0214023, -0.0124864, -0.0213927, -0.0122862, -0.0046058, 0.0044310
8: 0.9624710, 0.9880158, 0.9624981, 0.9885898, -0.0185705, 0.0186787
9: 0.0011753, 0.0086830, 0.0010066, 0.0086750, -0.0042078, 0.0042986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113831, upper bound: 0.0113831
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113831, upper bound: 0.0113842
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0004476, 0.0010348, -0.0004617, 0.0009898, -0.0010357, 0.0010869
1: -0.0007750, 0.0029076, -0.0008411, 0.0028386, -0.0029530, 0.0030720
2: 0.0119856, 0.0175007, 0.0120889, 0.0175996, -0.0042797, 0.0040847
3: -0.0016142, 0.0025329, -0.0015365, 0.0026072, -0.0030711, 0.0029155
4: -0.0058686, -0.0020433, -0.0057969, -0.0019747, -0.0038939, 0.0037536
5: 0.0063269, 0.0104664, 0.0064044, 0.0105407, -0.0030515, 0.0028953
6: 0.0074688, 0.0105453, 0.0075734, 0.0105160, -0.0030473, 0.0029719
7: -0.0211209, -0.0121345, -0.0212820, -0.0123029, -0.0046398, 0.0050378
8: 0.9632769, 0.9890243, 0.9628154, 0.9885417, -0.0194213, 0.0203116
9: 0.0008789, 0.0084461, 0.0010207, 0.0085817, -0.0047126, 0.0043936

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A1_B1_A2_A2_A1

### Relational analysis result of IS_A1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112622, upper bound: 0.0112744
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112736, upper bound: 0.0112736
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004626, 0.0010106, -0.0004965, 0.0012114, -0.0012556, 0.0011400
1: -0.0008452, 0.0028704, -0.0010041, 0.0031782, -0.0033304, 0.0029365
2: 0.0120413, 0.0176058, 0.0115803, 0.0178437, -0.0040957, 0.0046548
3: -0.0015724, 0.0026119, -0.0019190, 0.0027908, -0.0029516, 0.0033486
4: -0.0058300, -0.0019704, -0.0061497, -0.0018054, -0.0038623, 0.0041793
5: 0.0063687, 0.0105453, 0.0060227, 0.0107239, -0.0029344, 0.0033280
6: 0.0075252, 0.0105295, 0.0070583, 0.0106601, -0.0031349, 0.0034712
7: -0.0212921, -0.0122252, -0.0216798, -0.0114741, -0.0055462, 0.0054651
8: 0.9627863, 0.9887643, 0.9616759, 0.9909164, -0.0220749, 0.0194185
9: 0.0009553, 0.0085902, 0.0003228, 0.0089167, -0.0047902, 0.0051777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111480, upper bound: 0.0119914
time: 0.72 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111480, upper bound: 0.0120051
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004372, 0.0011030, -0.0004873, 0.0012072, -0.0012522, 0.0012340
1: -0.0007264, 0.0030120, -0.0009611, 0.0031717, -0.0034682, 0.0031409
2: 0.0118292, 0.0174279, 0.0115901, 0.0177793, -0.0044017, 0.0048291
3: -0.0017318, 0.0024781, -0.0019116, 0.0027424, -0.0031835, 0.0034628
4: -0.0059771, -0.0020938, -0.0061429, -0.0018500, -0.0040725, 0.0040491
5: 0.0062095, 0.0104118, 0.0060300, 0.0106756, -0.0031662, 0.0034403
6: 0.0073104, 0.0105896, 0.0070682, 0.0106573, -0.0033469, 0.0035214
7: -0.0210023, -0.0118797, -0.0215749, -0.0114900, -0.0055894, 0.0059675
8: 0.9636168, 0.9897543, 0.9619762, 0.9908708, -0.0229250, 0.0208464
9: 0.0006644, 0.0083462, 0.0003362, 0.0088284, -0.0052206, 0.0052940

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111480, upper bound: 0.0120051
time: 0.76 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111480, upper bound: 0.0120051
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0004714, 0.0009943, -0.0004983, 0.0011751, -0.0012407, 0.0011228
1: -0.0008864, 0.0028454, -0.0010123, 0.0031225, -0.0033292, 0.0028734
2: 0.0120787, 0.0176675, 0.0116637, 0.0178561, -0.0040140, 0.0046601
3: -0.0015442, 0.0026583, -0.0018563, 0.0028001, -0.0028952, 0.0033557
4: -0.0058040, -0.0019276, -0.0060918, -0.0017968, -0.0037551, 0.0041643
5: 0.0063968, 0.0105916, 0.0060853, 0.0107332, -0.0028783, 0.0033354
6: 0.0075630, 0.0105189, 0.0071428, 0.0106365, -0.0030734, 0.0033761
7: -0.0213927, -0.0122862, -0.0217000, -0.0116100, -0.0056226, 0.0053829
8: 0.9624981, 0.9885898, 0.9616177, 0.9905269, -0.0220927, 0.0190275
9: 0.0010066, 0.0086750, 0.0004373, 0.0089337, -0.0047134, 0.0052112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_B2_B1_A1

### Relational analysis result of IS_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111511, upper bound: 0.0119908
time: 0.63 seconds

## Relational analysis of IS_A1_B2_B2_B1_A2

### Relational analysis result of IS_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111511, upper bound: 0.0119908
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0004617, 0.0009898, -0.0004771, 0.0012833, -0.0013510, 0.0011149
1: -0.0008411, 0.0028386, -0.0009130, 0.0032884, -0.0035176, 0.0029884
2: 0.0120889, 0.0175996, 0.0114153, 0.0177073, -0.0041372, 0.0049470
3: -0.0015365, 0.0026072, -0.0020431, 0.0026883, -0.0029680, 0.0035729
4: -0.0057969, -0.0019747, -0.0062642, -0.0019000, -0.0038970, 0.0042895
5: 0.0064044, 0.0105407, 0.0058988, 0.0106215, -0.0029492, 0.0035523
6: 0.0075734, 0.0105160, 0.0068912, 0.0107069, -0.0031334, 0.0036249
7: -0.0212820, -0.0123029, -0.0214576, -0.0112051, -0.0061252, 0.0053475
8: 0.9628154, 0.9885417, 0.9623121, 0.9916869, -0.0234270, 0.0196577
9: 0.0010207, 0.0085817, 0.0000963, 0.0087296, -0.0047194, 0.0056282

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A1_B2_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110459, upper bound: 0.0118923
time: 0.63 seconds

## Relational analysis of IS_A1_B2_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110318, upper bound: 0.0118975
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004965, 0.0012114, -0.0004626, 0.0010106, -0.0011400, 0.0012556
1: -0.0010041, 0.0031782, -0.0008452, 0.0028704, -0.0029365, 0.0033304
2: 0.0115803, 0.0178437, 0.0120413, 0.0176058, -0.0046548, 0.0040957
3: -0.0019190, 0.0027908, -0.0015724, 0.0026119, -0.0033486, 0.0029516
4: -0.0061497, -0.0018054, -0.0058300, -0.0019704, -0.0041793, 0.0038623
5: 0.0060227, 0.0107239, 0.0063687, 0.0105453, -0.0033280, 0.0029344
6: 0.0070583, 0.0106601, 0.0075252, 0.0105295, -0.0034712, 0.0031349
7: -0.0216798, -0.0114741, -0.0212921, -0.0122252, -0.0054651, 0.0055462
8: 0.9616759, 0.9909164, 0.9627863, 0.9887643, -0.0194185, 0.0220749
9: 0.0003228, 0.0089167, 0.0009553, 0.0085902, -0.0051777, 0.0047902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119914, upper bound: 0.0111480
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119914, upper bound: 0.0111480
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004873, 0.0012072, -0.0004372, 0.0011030, -0.0012340, 0.0012522
1: -0.0009611, 0.0031717, -0.0007264, 0.0030120, -0.0031409, 0.0034682
2: 0.0115901, 0.0177793, 0.0118292, 0.0174279, -0.0048291, 0.0044017
3: -0.0019116, 0.0027424, -0.0017318, 0.0024781, -0.0034628, 0.0031835
4: -0.0061429, -0.0018500, -0.0059771, -0.0020938, -0.0040491, 0.0040725
5: 0.0060300, 0.0106756, 0.0062095, 0.0104118, -0.0034403, 0.0031662
6: 0.0070682, 0.0106573, 0.0073104, 0.0105896, -0.0035214, 0.0033469
7: -0.0215749, -0.0114900, -0.0210023, -0.0118797, -0.0059675, 0.0055894
8: 0.9619762, 0.9908708, 0.9636168, 0.9897543, -0.0208464, 0.0229250
9: 0.0003362, 0.0088284, 0.0006644, 0.0083462, -0.0052940, 0.0052206

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A1_B2_B1

### Relational analysis result of IS_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120051, upper bound: 0.0111480
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B2_B2

### Relational analysis result of IS_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120051, upper bound: 0.0111480
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0004983, 0.0011751, -0.0004714, 0.0009943, -0.0011228, 0.0012407
1: -0.0010123, 0.0031225, -0.0008864, 0.0028454, -0.0028734, 0.0033292
2: 0.0116637, 0.0178561, 0.0120787, 0.0176675, -0.0046601, 0.0040140
3: -0.0018563, 0.0028001, -0.0015442, 0.0026583, -0.0033557, 0.0028952
4: -0.0060918, -0.0017968, -0.0058040, -0.0019276, -0.0041643, 0.0037551
5: 0.0060853, 0.0107332, 0.0063968, 0.0105916, -0.0033354, 0.0028783
6: 0.0071428, 0.0106365, 0.0075630, 0.0105189, -0.0033761, 0.0030734
7: -0.0217000, -0.0116100, -0.0213927, -0.0122862, -0.0053829, 0.0056226
8: 0.9616177, 0.9905269, 0.9624981, 0.9885898, -0.0190275, 0.0220927
9: 0.0004373, 0.0089337, 0.0010066, 0.0086750, -0.0052112, 0.0047134

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119908, upper bound: 0.0111511
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_A1_B2

### Relational analysis result of IS_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119908, upper bound: 0.0111511
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0004771, 0.0012833, -0.0004617, 0.0009898, -0.0011149, 0.0013510
1: -0.0009130, 0.0032884, -0.0008411, 0.0028386, -0.0029884, 0.0035176
2: 0.0114153, 0.0177073, 0.0120889, 0.0175996, -0.0049470, 0.0041372
3: -0.0020431, 0.0026883, -0.0015365, 0.0026072, -0.0035729, 0.0029680
4: -0.0062642, -0.0019000, -0.0057969, -0.0019747, -0.0042895, 0.0038970
5: 0.0058988, 0.0106215, 0.0064044, 0.0105407, -0.0035523, 0.0029492
6: 0.0068912, 0.0107069, 0.0075734, 0.0105160, -0.0036249, 0.0031334
7: -0.0214576, -0.0112051, -0.0212820, -0.0123029, -0.0053475, 0.0061252
8: 0.9623121, 0.9916869, 0.9628154, 0.9885417, -0.0196577, 0.0234270
9: 0.0000963, 0.0087296, 0.0010207, 0.0085817, -0.0056282, 0.0047194

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A2_B1_A2_A2_A1

### Relational analysis result of IS_A2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118923, upper bound: 0.0110459
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118975, upper bound: 0.0110318
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0004870, 0.0012076, -0.0004975, 0.0012458, -0.0012914, 0.0012652
1: -0.0009594, 0.0031724, -0.0010086, 0.0032309, -0.0032121, 0.0032505
2: 0.0115890, 0.0177767, 0.0115014, 0.0178505, -0.0044783, 0.0044342
3: -0.0019124, 0.0027405, -0.0019783, 0.0027959, -0.0031893, 0.0031629
4: -0.0061437, -0.0018518, -0.0062044, -0.0018007, -0.0043126, 0.0042425
5: 0.0060292, 0.0106736, 0.0059634, 0.0107290, -0.0031665, 0.0031408
6: 0.0070672, 0.0106576, 0.0069784, 0.0106825, -0.0036153, 0.0036792
7: -0.0215707, -0.0114883, -0.0216909, -0.0113455, -0.0053513, 0.0053690
8: 0.9619883, 0.9908757, 0.9616439, 0.9912848, -0.0210909, 0.0213104
9: 0.0003348, 0.0088248, 0.0002145, 0.0089260, -0.0048413, 0.0048182

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A1_A1_B1

### Relational analysis result of IS_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119914, upper bound: 0.0111580
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2

### Relational analysis result of IS_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119914, upper bound: 0.0111585
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0004653, 0.0013117, -0.0004883, 0.0012416, -0.0012850, 0.0013720
1: -0.0008580, 0.0033318, -0.0009657, 0.0032244, -0.0033588, 0.0034548
2: 0.0113502, 0.0176249, 0.0115111, 0.0177863, -0.0047857, 0.0046108
3: -0.0020920, 0.0026263, -0.0019711, 0.0027476, -0.0034211, 0.0032783
4: -0.0063093, -0.0019571, -0.0061977, -0.0018452, -0.0044641, 0.0042406
5: 0.0058500, 0.0105597, 0.0059707, 0.0106808, -0.0033978, 0.0032544
6: 0.0068253, 0.0107253, 0.0069882, 0.0106797, -0.0038544, 0.0037371
7: -0.0213233, -0.0110992, -0.0215862, -0.0113612, -0.0053432, 0.0058829
8: 0.9626971, 0.9919905, 0.9619439, 0.9912397, -0.0219589, 0.0227435
9: 0.0000071, 0.0086165, 0.0002278, 0.0088379, -0.0052808, 0.0048946

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A1_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120051, upper bound: 0.0111585
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120051, upper bound: 0.0111585
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0004983, 0.0011751, -0.0004970, 0.0012259, -0.0012869, 0.0012410
1: -0.0010123, 0.0031225, -0.0010066, 0.0032003, -0.0031771, 0.0032010
2: 0.0116637, 0.0178561, 0.0115472, 0.0178475, -0.0044114, 0.0044021
3: -0.0018563, 0.0028001, -0.0019439, 0.0027937, -0.0031423, 0.0031472
4: -0.0060918, -0.0017968, -0.0061727, -0.0018027, -0.0042443, 0.0041668
5: 0.0060853, 0.0107332, 0.0059978, 0.0107267, -0.0031199, 0.0031262
6: 0.0071428, 0.0106365, 0.0070247, 0.0106695, -0.0035267, 0.0036117
7: -0.0217000, -0.0116100, -0.0216860, -0.0114200, -0.0054565, 0.0053200
8: 0.9616177, 0.9905269, 0.9616579, 0.9910712, -0.0209217, 0.0209904
9: 0.0004373, 0.0089337, 0.0002773, 0.0089219, -0.0047851, 0.0048753

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A2_A1_B1

### Relational analysis result of IS_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119908, upper bound: 0.0111647
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2

### Relational analysis result of IS_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119908, upper bound: 0.0111647
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0004771, 0.0012833, -0.0004879, 0.0012217, -0.0012792, 0.0013481
1: -0.0009130, 0.0032884, -0.0009637, 0.0031939, -0.0033199, 0.0034014
2: 0.0114153, 0.0177073, 0.0115568, 0.0177832, -0.0047130, 0.0045727
3: -0.0020431, 0.0026883, -0.0019366, 0.0027453, -0.0033698, 0.0032602
4: -0.0062642, -0.0019000, -0.0061660, -0.0018474, -0.0044168, 0.0042660
5: 0.0058988, 0.0106215, 0.0060050, 0.0106785, -0.0033469, 0.0032371
6: 0.0068912, 0.0107069, 0.0070345, 0.0106668, -0.0037756, 0.0036723
7: -0.0214576, -0.0112051, -0.0215812, -0.0114358, -0.0054144, 0.0058246
8: 0.9623121, 0.9916869, 0.9619581, 0.9910260, -0.0217572, 0.0223980
9: 0.0000963, 0.0087296, 0.0002906, 0.0088337, -0.0052175, 0.0049367

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A2_B2_A2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118923, upper bound: 0.0110588
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118975, upper bound: 0.0110433
time: 0.79 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.90 seconds
IS_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 8, lower bound: -0.0113869, upper bound: 0.0113935
IS_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 8, lower bound: -0.0113869, upper bound: 0.0113947
IS_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 8, lower bound: -0.0113876, upper bound: 0.0113947
IS_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 8, lower bound: -0.0113876, upper bound: 0.0113947
IS_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 8, lower bound: -0.0113831, upper bound: 0.0113831
IS_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 8, lower bound: -0.0113831, upper bound: 0.0113842
IS_A1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 8, lower bound: -0.0112622, upper bound: 0.0112744
IS_A1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 8, lower bound: -0.0112736, upper bound: 0.0112736
IS_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 8, lower bound: -0.0111480, upper bound: 0.0119914
IS_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 8, lower bound: -0.0111480, upper bound: 0.0120051
IS_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 8, lower bound: -0.0111480, upper bound: 0.0120051
IS_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 8, lower bound: -0.0111480, upper bound: 0.0120051
IS_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 8, lower bound: -0.0111511, upper bound: 0.0119908
IS_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 8, lower bound: -0.0111511, upper bound: 0.0119908
IS_A1_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 8, lower bound: -0.0110459, upper bound: 0.0118923
IS_A1_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 8, lower bound: -0.0110318, upper bound: 0.0118975
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 8, lower bound: -0.0119914, upper bound: 0.0111480
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 8, lower bound: -0.0119914, upper bound: 0.0111480
IS_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 8, lower bound: -0.0120051, upper bound: 0.0111480
IS_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 8, lower bound: -0.0120051, upper bound: 0.0111480
IS_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 8, lower bound: -0.0119908, upper bound: 0.0111511
IS_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 8, lower bound: -0.0119908, upper bound: 0.0111511
IS_A2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 8, lower bound: -0.0118923, upper bound: 0.0110459
IS_A2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 8, lower bound: -0.0118975, upper bound: 0.0110318
IS_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 8, lower bound: -0.0119914, upper bound: 0.0111580
IS_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 8, lower bound: -0.0119914, upper bound: 0.0111585
IS_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 8, lower bound: -0.0120051, upper bound: 0.0111585
IS_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 8, lower bound: -0.0120051, upper bound: 0.0111585
IS_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 8, lower bound: -0.0119908, upper bound: 0.0111647
IS_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 8, lower bound: -0.0119908, upper bound: 0.0111647
IS_A2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 8, lower bound: -0.0118923, upper bound: 0.0110588
IS_A2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 8, lower bound: -0.0118975, upper bound: 0.0110433

## BFS IS instance: IS_A1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004619, 0.0009759, -0.0004626, 0.0010106, -0.0010349, 0.0010028
1: -0.0008421, 0.0028172, -0.0008452, 0.0028704, -0.0028308, 0.0028363
2: 0.0121210, 0.0176011, 0.0120413, 0.0176058, -0.0039148, 0.0039188
3: -0.0015124, 0.0026084, -0.0015724, 0.0026119, -0.0027922, 0.0028010
4: -0.0057747, -0.0019736, -0.0058300, -0.0019704, -0.0037150, 0.0036820
5: 0.0064285, 0.0105418, 0.0063687, 0.0105453, -0.0027726, 0.0027819
6: 0.0076059, 0.0105070, 0.0075252, 0.0105295, -0.0029237, 0.0029818
7: -0.0212845, -0.0123551, -0.0212921, -0.0122252, -0.0044352, 0.0043405
8: 0.9628081, 0.9883922, 0.9627863, 0.9887643, -0.0186271, 0.0186203
9: 0.0010647, 0.0085839, 0.0009553, 0.0085902, -0.0041625, 0.0042143

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114056, upper bound: 0.0114012
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114056, upper bound: 0.0114012
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004619, 0.0009759, -0.0004372, 0.0011030, -0.0011362, 0.0010042
1: -0.0008421, 0.0028172, -0.0007264, 0.0030120, -0.0030962, 0.0028785
2: 0.0121210, 0.0176011, 0.0118292, 0.0174279, -0.0039537, 0.0043163
3: -0.0015124, 0.0026084, -0.0017318, 0.0024781, -0.0028123, 0.0030999
4: -0.0057747, -0.0019736, -0.0059771, -0.0020938, -0.0036809, 0.0039577
5: 0.0064285, 0.0105418, 0.0062095, 0.0104118, -0.0027919, 0.0030802
6: 0.0076059, 0.0105070, 0.0073104, 0.0105896, -0.0029837, 0.0031965
7: -0.0212845, -0.0123551, -0.0210023, -0.0118797, -0.0050828, 0.0043771
8: 0.9628081, 0.9883922, 0.9636168, 0.9897543, -0.0204826, 0.0188247
9: 0.0010647, 0.0085839, 0.0006644, 0.0083462, -0.0041821, 0.0047596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114056, upper bound: 0.0114012
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114056, upper bound: 0.0114012
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004366, 0.0010694, -0.0004613, 0.0009754, -0.0010006, 0.0011086
1: -0.0007235, 0.0029606, -0.0008392, 0.0028165, -0.0029157, 0.0030458
2: 0.0119062, 0.0174235, 0.0121220, 0.0175968, -0.0042442, 0.0040129
3: -0.0016739, 0.0024749, -0.0015117, 0.0026052, -0.0030471, 0.0028545
4: -0.0059237, -0.0020968, -0.0057740, -0.0019766, -0.0039044, 0.0036772
5: 0.0062673, 0.0104085, 0.0064292, 0.0105386, -0.0030277, 0.0028338
6: 0.0073884, 0.0105678, 0.0076069, 0.0105067, -0.0031183, 0.0029609
7: -0.0209952, -0.0120051, -0.0212776, -0.0123567, -0.0043458, 0.0050229
8: 0.9636371, 0.9893949, 0.9628282, 0.9883875, -0.0191025, 0.0201433
9: 0.0007700, 0.0083402, 0.0010660, 0.0085780, -0.0046852, 0.0042190

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A1_B1_A1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112659, upper bound: 0.0112853
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112781, upper bound: 0.0112842
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004366, 0.0010694, -0.0004715, 0.0009402, -0.0009686, 0.0011281
1: -0.0007235, 0.0029606, -0.0008872, 0.0027626, -0.0029079, 0.0030451
2: 0.0119062, 0.0174235, 0.0122028, 0.0176687, -0.0042592, 0.0040013
3: -0.0016739, 0.0024749, -0.0014509, 0.0026592, -0.0030647, 0.0028457
4: -0.0059237, -0.0020968, -0.0057180, -0.0019268, -0.0038689, 0.0036212
5: 0.0062673, 0.0104085, 0.0064899, 0.0105926, -0.0030460, 0.0028250
6: 0.0073884, 0.0105678, 0.0076887, 0.0104838, -0.0030954, 0.0028791
7: -0.0209952, -0.0120051, -0.0213947, -0.0124883, -0.0043268, 0.0052147
8: 0.9636371, 0.9893949, 0.9624925, 0.9880104, -0.0190479, 0.0201980
9: 0.0007700, 0.0083402, 0.0011769, 0.0086766, -0.0047913, 0.0042030

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A1_B1_A1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112659, upper bound: 0.0112853
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112781, upper bound: 0.0112842
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004722, 0.0009407, -0.0004623, 0.0009901, -0.0010366, 0.0009769
1: -0.0008903, 0.0027633, -0.0008439, 0.0028391, -0.0028082, 0.0027974
2: 0.0122016, 0.0176734, 0.0120882, 0.0176038, -0.0038642, 0.0039029
3: -0.0014518, 0.0026627, -0.0015371, 0.0026105, -0.0027577, 0.0027965
4: -0.0057188, -0.0019235, -0.0057975, -0.0019718, -0.0036593, 0.0036249
5: 0.0064890, 0.0105960, 0.0064039, 0.0105439, -0.0027385, 0.0027782
6: 0.0076875, 0.0104841, 0.0075726, 0.0105163, -0.0028287, 0.0029115
7: -0.0214023, -0.0124864, -0.0212890, -0.0123016, -0.0045930, 0.0043071
8: 0.9624710, 0.9880158, 0.9627954, 0.9885455, -0.0185338, 0.0183753
9: 0.0011753, 0.0086830, 0.0010196, 0.0085876, -0.0041192, 0.0042878

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A1_B1_A2_A1_B1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112949, upper bound: 0.0112711
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112956, upper bound: 0.0112860
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004722, 0.0009407, -0.0004369, 0.0010826, -0.0011389, 0.0009783
1: -0.0008903, 0.0027633, -0.0007250, 0.0029807, -0.0030688, 0.0028396
2: 0.0122016, 0.0176734, 0.0118760, 0.0174258, -0.0039043, 0.0042932
3: -0.0014518, 0.0026627, -0.0016967, 0.0024766, -0.0027782, 0.0030900
4: -0.0057188, -0.0019235, -0.0059446, -0.0020952, -0.0036235, 0.0038956
5: 0.0064890, 0.0105960, 0.0062446, 0.0104102, -0.0027582, 0.0030712
6: 0.0076875, 0.0104841, 0.0073578, 0.0105764, -0.0028888, 0.0031264
7: -0.0214023, -0.0124864, -0.0209989, -0.0119559, -0.0052291, 0.0043468
8: 0.9624710, 0.9880158, 0.9636264, 0.9895360, -0.0203561, 0.0185860
9: 0.0011753, 0.0086830, 0.0007285, 0.0083433, -0.0041410, 0.0048234

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A1_B1_A2_A1_B2_B1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112949, upper bound: 0.0112711
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112956, upper bound: 0.0112860
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0004518, 0.0010047, -0.0004615, 0.0009801, -0.0010211, 0.0010557
1: -0.0007948, 0.0028614, -0.0008401, 0.0028237, -0.0028867, 0.0030009
2: 0.0120548, 0.0175302, 0.0121112, 0.0175981, -0.0041768, 0.0039913
3: -0.0015622, 0.0025551, -0.0015198, 0.0026061, -0.0029952, 0.0028487
4: -0.0058206, -0.0020228, -0.0057815, -0.0019757, -0.0038449, 0.0037587
5: 0.0063788, 0.0104886, 0.0064212, 0.0105395, -0.0029759, 0.0028292
6: 0.0075388, 0.0105257, 0.0075960, 0.0105097, -0.0029709, 0.0029297
7: -0.0211691, -0.0122471, -0.0212796, -0.0123392, -0.0045163, 0.0048916
8: 0.9631390, 0.9887015, 0.9628222, 0.9884378, -0.0189791, 0.0198282
9: 0.0009738, 0.0084866, 0.0010513, 0.0085797, -0.0045837, 0.0042961

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A2_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112622, upper bound: 0.0112740
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_A2_A1_B2

### Relational analysis result of IS_A1_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112622, upper bound: 0.0112740
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0004472, 0.0010101, -0.0004616, 0.0009839, -0.0010297, 0.0010560
1: -0.0007732, 0.0028696, -0.0008406, 0.0028294, -0.0029070, 0.0029989
2: 0.0120425, 0.0174979, 0.0121026, 0.0175989, -0.0041694, 0.0040247
3: -0.0015715, 0.0025308, -0.0015263, 0.0026067, -0.0029870, 0.0028748
4: -0.0058291, -0.0020452, -0.0057875, -0.0019752, -0.0038540, 0.0037423
5: 0.0063696, 0.0104644, 0.0064147, 0.0105402, -0.0029674, 0.0028551
6: 0.0075264, 0.0105292, 0.0075872, 0.0105122, -0.0029858, 0.0029420
7: -0.0211164, -0.0122271, -0.0212810, -0.0123251, -0.0046072, 0.0048495
8: 0.9632899, 0.9887587, 0.9628184, 0.9884782, -0.0191315, 0.0197991
9: 0.0009570, 0.0084423, 0.0010394, 0.0085809, -0.0045479, 0.0043471

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A2_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112736, upper bound: 0.0112736
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112736, upper bound: 0.0112736
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004626, 0.0010106, -0.0004870, 0.0012076, -0.0012518, 0.0011291
1: -0.0008452, 0.0028704, -0.0009594, 0.0031724, -0.0033249, 0.0028981
2: 0.0120413, 0.0176058, 0.0115890, 0.0177767, -0.0040340, 0.0046465
3: -0.0015724, 0.0026119, -0.0019124, 0.0027405, -0.0029049, 0.0033424
4: -0.0058300, -0.0019704, -0.0061437, -0.0018518, -0.0038190, 0.0041733
5: 0.0063687, 0.0105453, 0.0060292, 0.0106736, -0.0028877, 0.0033218
6: 0.0075252, 0.0105295, 0.0070672, 0.0106576, -0.0031325, 0.0034624
7: -0.0212921, -0.0122252, -0.0215707, -0.0114883, -0.0055327, 0.0053398
8: 0.9627863, 0.9887643, 0.9619883, 0.9908757, -0.0220362, 0.0191339
9: 0.0009553, 0.0085902, 0.0003348, 0.0088248, -0.0046872, 0.0051664

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111856, upper bound: 0.0120231
time: 0.78 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111856, upper bound: 0.0120282
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004626, 0.0010106, -0.0004653, 0.0013117, -0.0013677, 0.0011203
1: -0.0008452, 0.0028704, -0.0008580, 0.0033318, -0.0035541, 0.0029501
2: 0.0120413, 0.0176058, 0.0113502, 0.0176249, -0.0040716, 0.0049898
3: -0.0015724, 0.0026119, -0.0020920, 0.0026263, -0.0029192, 0.0036006
4: -0.0058300, -0.0019704, -0.0063093, -0.0019571, -0.0038728, 0.0043389
5: 0.0063687, 0.0105453, 0.0058500, 0.0105597, -0.0029009, 0.0035795
6: 0.0075252, 0.0105295, 0.0068253, 0.0107253, -0.0032001, 0.0037042
7: -0.0212921, -0.0122252, -0.0213233, -0.0110992, -0.0060921, 0.0052477
8: 0.9627863, 0.9887643, 0.9626971, 0.9919905, -0.0236390, 0.0193482
9: 0.0009553, 0.0085902, 0.0000071, 0.0086165, -0.0046626, 0.0056374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111856, upper bound: 0.0120239
time: 0.76 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111856, upper bound: 0.0120282
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0004366, 0.0010694, -0.0004873, 0.0012072, -0.0012496, 0.0012001
1: -0.0007235, 0.0029606, -0.0009611, 0.0031717, -0.0034042, 0.0030935
2: 0.0119062, 0.0174235, 0.0115901, 0.0177793, -0.0043306, 0.0047446
3: -0.0016739, 0.0024749, -0.0019116, 0.0027424, -0.0031300, 0.0034047
4: -0.0059237, -0.0020968, -0.0061429, -0.0018500, -0.0040232, 0.0040461
5: 0.0062673, 0.0104085, 0.0060300, 0.0106756, -0.0031128, 0.0033830
6: 0.0073884, 0.0105678, 0.0070682, 0.0106573, -0.0032690, 0.0034996
7: -0.0209952, -0.0120051, -0.0215749, -0.0114900, -0.0055380, 0.0058516
8: 0.9636371, 0.9893949, 0.9619762, 0.9908708, -0.0225182, 0.0205145
9: 0.0007700, 0.0083402, 0.0003362, 0.0088284, -0.0051231, 0.0052229

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A1_B2_B1_A2_A1_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110084, upper bound: 0.0118954
time: 0.79 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_A2

### Relational analysis result of IS_A1_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110280, upper bound: 0.0119031
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0004476, 0.0010348, -0.0004873, 0.0012072, -0.0012705, 0.0011686
1: -0.0007750, 0.0029076, -0.0009611, 0.0031717, -0.0034029, 0.0030351
2: 0.0119856, 0.0175007, 0.0115901, 0.0177793, -0.0042431, 0.0047585
3: -0.0016142, 0.0025329, -0.0019116, 0.0027424, -0.0030642, 0.0034221
4: -0.0058686, -0.0020433, -0.0061429, -0.0018500, -0.0039625, 0.0040996
5: 0.0063269, 0.0104664, 0.0060300, 0.0106756, -0.0030471, 0.0034011
6: 0.0074688, 0.0105453, 0.0070682, 0.0106573, -0.0031886, 0.0034771
7: -0.0211209, -0.0121345, -0.0215749, -0.0114900, -0.0057377, 0.0057091
8: 0.9632769, 0.9890243, 0.9619762, 0.9908708, -0.0225670, 0.0201062
9: 0.0008789, 0.0084461, 0.0003362, 0.0088284, -0.0050031, 0.0053181

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A1_B2_B1_A2_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110084, upper bound: 0.0118954
time: 0.75 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110280, upper bound: 0.0119031
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004623, 0.0009901, -0.0004983, 0.0011751, -0.0012291, 0.0011186
1: -0.0008439, 0.0028391, -0.0010123, 0.0031225, -0.0032858, 0.0028675
2: 0.0120882, 0.0176038, 0.0116637, 0.0178561, -0.0040052, 0.0045955
3: -0.0015371, 0.0026105, -0.0018563, 0.0028001, -0.0028886, 0.0033076
4: -0.0057975, -0.0019718, -0.0060918, -0.0017968, -0.0037490, 0.0041201
5: 0.0064039, 0.0105439, 0.0060853, 0.0107332, -0.0028716, 0.0032874
6: 0.0075726, 0.0105163, 0.0071428, 0.0106365, -0.0030639, 0.0033735
7: -0.0212890, -0.0123016, -0.0217000, -0.0116100, -0.0054987, 0.0053686
8: 0.9627954, 0.9885455, 0.9616177, 0.9905269, -0.0217893, 0.0189864
9: 0.0010196, 0.0085876, 0.0004373, 0.0089337, -0.0047013, 0.0051226

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A1_B2_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110190, upper bound: 0.0118806
time: 0.73 seconds

## Relational analysis of IS_A1_B2_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110321, upper bound: 0.0118857
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004369, 0.0010826, -0.0004983, 0.0011751, -0.0012305, 0.0012204
1: -0.0007250, 0.0029807, -0.0010123, 0.0031225, -0.0033279, 0.0031043
2: 0.0118760, 0.0174258, 0.0116637, 0.0178561, -0.0043598, 0.0046355
3: -0.0016967, 0.0024766, -0.0018563, 0.0028001, -0.0031552, 0.0033281
4: -0.0059446, -0.0020952, -0.0060918, -0.0017968, -0.0039949, 0.0039966
5: 0.0062446, 0.0104102, 0.0060853, 0.0107332, -0.0031378, 0.0033071
6: 0.0073578, 0.0105764, 0.0071428, 0.0106365, -0.0032787, 0.0034336
7: -0.0209989, -0.0119559, -0.0217000, -0.0116100, -0.0055384, 0.0059463
8: 0.9636264, 0.9895360, 0.9616177, 0.9905269, -0.0220000, 0.0206415
9: 0.0007285, 0.0083433, 0.0004373, 0.0089337, -0.0051878, 0.0051444

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A1_B2_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110190, upper bound: 0.0118806
time: 0.81 seconds

## Relational analysis of IS_A1_B2_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110321, upper bound: 0.0118857
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0004615, 0.0009801, -0.0004783, 0.0012478, -0.0013144, 0.0011034
1: -0.0008401, 0.0028237, -0.0009188, 0.0032339, -0.0034429, 0.0029256
2: 0.0121112, 0.0175981, 0.0114969, 0.0177161, -0.0040550, 0.0048387
3: -0.0015198, 0.0026061, -0.0019817, 0.0026948, -0.0029112, 0.0034930
4: -0.0057815, -0.0019757, -0.0062076, -0.0018939, -0.0038725, 0.0042319
5: 0.0064212, 0.0105395, 0.0059600, 0.0106281, -0.0028929, 0.0034727
6: 0.0075960, 0.0105097, 0.0069738, 0.0106837, -0.0030878, 0.0035359
7: -0.0212796, -0.0123392, -0.0214718, -0.0113381, -0.0059702, 0.0052825
8: 0.9628222, 0.9884378, 0.9622716, 0.9913060, -0.0229184, 0.0192536
9: 0.0010513, 0.0085797, 0.0002083, 0.0087416, -0.0046439, 0.0054920

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_B2_B2_B1_A1

### Relational analysis result of IS_A1_B2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110459, upper bound: 0.0118923
time: 0.65 seconds

## Relational analysis of IS_A1_B2_B2_B2_B1_A2

### Relational analysis result of IS_A1_B2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110459, upper bound: 0.0118761
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0004616, 0.0009839, -0.0004766, 0.0012603, -0.0013195, 0.0011086
1: -0.0008406, 0.0028294, -0.0009108, 0.0032530, -0.0034407, 0.0029567
2: 0.0121026, 0.0175989, 0.0114682, 0.0177040, -0.0040967, 0.0048311
3: -0.0015263, 0.0026067, -0.0020033, 0.0026858, -0.0029400, 0.0034846
4: -0.0057875, -0.0019752, -0.0062275, -0.0019023, -0.0038852, 0.0042523
5: 0.0064147, 0.0105402, 0.0059385, 0.0106190, -0.0029217, 0.0034640
6: 0.0075872, 0.0105122, 0.0069448, 0.0106918, -0.0031046, 0.0035673
7: -0.0212810, -0.0123251, -0.0214522, -0.0112915, -0.0059277, 0.0053209
8: 0.9628184, 0.9884782, 0.9623277, 0.9914396, -0.0228882, 0.0194558
9: 0.0010394, 0.0085809, 0.0001690, 0.0087251, -0.0046900, 0.0054558

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_B2_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110318, upper bound: 0.0118975
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B2_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110318, upper bound: 0.0118845
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004870, 0.0012076, -0.0004626, 0.0010106, -0.0011291, 0.0012518
1: -0.0009594, 0.0031724, -0.0008452, 0.0028704, -0.0028981, 0.0033249
2: 0.0115890, 0.0177767, 0.0120413, 0.0176058, -0.0046465, 0.0040340
3: -0.0019124, 0.0027405, -0.0015724, 0.0026119, -0.0033424, 0.0029049
4: -0.0061437, -0.0018518, -0.0058300, -0.0019704, -0.0041733, 0.0038190
5: 0.0060292, 0.0106736, 0.0063687, 0.0105453, -0.0033218, 0.0028877
6: 0.0070672, 0.0106576, 0.0075252, 0.0105295, -0.0034624, 0.0031325
7: -0.0215707, -0.0114883, -0.0212921, -0.0122252, -0.0053398, 0.0055327
8: 0.9619883, 0.9908757, 0.9627863, 0.9887643, -0.0191339, 0.0220362
9: 0.0003348, 0.0088248, 0.0009553, 0.0085902, -0.0051664, 0.0046872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120230, upper bound: 0.0111856
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120230, upper bound: 0.0111856
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004653, 0.0013117, -0.0004626, 0.0010106, -0.0011203, 0.0013677
1: -0.0008580, 0.0033318, -0.0008452, 0.0028704, -0.0029501, 0.0035541
2: 0.0113502, 0.0176249, 0.0120413, 0.0176058, -0.0049898, 0.0040716
3: -0.0020920, 0.0026263, -0.0015724, 0.0026119, -0.0036006, 0.0029192
4: -0.0063093, -0.0019571, -0.0058300, -0.0019704, -0.0043389, 0.0038728
5: 0.0058500, 0.0105597, 0.0063687, 0.0105453, -0.0035795, 0.0029009
6: 0.0068253, 0.0107253, 0.0075252, 0.0105295, -0.0037042, 0.0032001
7: -0.0213233, -0.0110992, -0.0212921, -0.0122252, -0.0052477, 0.0060921
8: 0.9626971, 0.9919905, 0.9627863, 0.9887643, -0.0193482, 0.0236390
9: 0.0000071, 0.0086165, 0.0009553, 0.0085902, -0.0056374, 0.0046626

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120230, upper bound: 0.0111856
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120230, upper bound: 0.0111856
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0004873, 0.0012072, -0.0004366, 0.0010694, -0.0012001, 0.0012496
1: -0.0009611, 0.0031717, -0.0007235, 0.0029606, -0.0030935, 0.0034042
2: 0.0115901, 0.0177793, 0.0119062, 0.0174235, -0.0047446, 0.0043306
3: -0.0019116, 0.0027424, -0.0016739, 0.0024749, -0.0034047, 0.0031300
4: -0.0061429, -0.0018500, -0.0059237, -0.0020968, -0.0040461, 0.0040232
5: 0.0060300, 0.0106756, 0.0062673, 0.0104085, -0.0033830, 0.0031128
6: 0.0070682, 0.0106573, 0.0073884, 0.0105678, -0.0034996, 0.0032690
7: -0.0215749, -0.0114900, -0.0209952, -0.0120051, -0.0058516, 0.0055380
8: 0.9619762, 0.9908708, 0.9636371, 0.9893949, -0.0205145, 0.0225182
9: 0.0003362, 0.0088284, 0.0007700, 0.0083402, -0.0052229, 0.0051231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A2_B1_A1_B2_B1_B1

### Relational analysis result of IS_A2_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118954, upper bound: 0.0110084
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B2_B1_B2

### Relational analysis result of IS_A2_B1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119031, upper bound: 0.0110280
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0004873, 0.0012072, -0.0004476, 0.0010348, -0.0011686, 0.0012705
1: -0.0009611, 0.0031717, -0.0007750, 0.0029076, -0.0030351, 0.0034029
2: 0.0115901, 0.0177793, 0.0119856, 0.0175007, -0.0047585, 0.0042431
3: -0.0019116, 0.0027424, -0.0016142, 0.0025329, -0.0034221, 0.0030642
4: -0.0061429, -0.0018500, -0.0058686, -0.0020433, -0.0040996, 0.0039625
5: 0.0060300, 0.0106756, 0.0063269, 0.0104664, -0.0034011, 0.0030471
6: 0.0070682, 0.0106573, 0.0074688, 0.0105453, -0.0034771, 0.0031886
7: -0.0215749, -0.0114900, -0.0211209, -0.0121345, -0.0057091, 0.0057377
8: 0.9619762, 0.9908708, 0.9632769, 0.9890243, -0.0201062, 0.0225670
9: 0.0003362, 0.0088284, 0.0008789, 0.0084461, -0.0053181, 0.0050031

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A2_B1_A1_B2_B2_B1

### Relational analysis result of IS_A2_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118954, upper bound: 0.0110084
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B2_B2_B2

### Relational analysis result of IS_A2_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119031, upper bound: 0.0110280
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004983, 0.0011751, -0.0004623, 0.0009901, -0.0011186, 0.0012291
1: -0.0010123, 0.0031225, -0.0008439, 0.0028391, -0.0028675, 0.0032858
2: 0.0116637, 0.0178561, 0.0120882, 0.0176038, -0.0045955, 0.0040052
3: -0.0018563, 0.0028001, -0.0015371, 0.0026105, -0.0033076, 0.0028886
4: -0.0060918, -0.0017968, -0.0057975, -0.0019718, -0.0041201, 0.0037490
5: 0.0060853, 0.0107332, 0.0064039, 0.0105439, -0.0032874, 0.0028716
6: 0.0071428, 0.0106365, 0.0075726, 0.0105163, -0.0033735, 0.0030639
7: -0.0217000, -0.0116100, -0.0212890, -0.0123016, -0.0053686, 0.0054987
8: 0.9616177, 0.9905269, 0.9627954, 0.9885455, -0.0189864, 0.0217893
9: 0.0004373, 0.0089337, 0.0010196, 0.0085876, -0.0051226, 0.0047013

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A2_B1_A2_A1_B1_B1

### Relational analysis result of IS_A2_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118806, upper bound: 0.0110190
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A2_A1_B1_B2

### Relational analysis result of IS_A2_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118857, upper bound: 0.0110321
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004983, 0.0011751, -0.0004369, 0.0010826, -0.0012204, 0.0012305
1: -0.0010123, 0.0031225, -0.0007250, 0.0029807, -0.0031043, 0.0033279
2: 0.0116637, 0.0178561, 0.0118760, 0.0174258, -0.0046355, 0.0043598
3: -0.0018563, 0.0028001, -0.0016967, 0.0024766, -0.0033281, 0.0031552
4: -0.0060918, -0.0017968, -0.0059446, -0.0020952, -0.0039966, 0.0039949
5: 0.0060853, 0.0107332, 0.0062446, 0.0104102, -0.0033071, 0.0031378
6: 0.0071428, 0.0106365, 0.0073578, 0.0105764, -0.0034336, 0.0032787
7: -0.0217000, -0.0116100, -0.0209989, -0.0119559, -0.0059463, 0.0055384
8: 0.9616177, 0.9905269, 0.9636264, 0.9895360, -0.0206415, 0.0220000
9: 0.0004373, 0.0089337, 0.0007285, 0.0083433, -0.0051444, 0.0051878

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A2_B1_A2_A1_B2_B1

### Relational analysis result of IS_A2_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118806, upper bound: 0.0110190
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_A1_B2_B2

### Relational analysis result of IS_A2_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118857, upper bound: 0.0110321
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0004783, 0.0012478, -0.0004615, 0.0009801, -0.0011034, 0.0013144
1: -0.0009188, 0.0032339, -0.0008401, 0.0028237, -0.0029256, 0.0034429
2: 0.0114969, 0.0177161, 0.0121112, 0.0175981, -0.0048387, 0.0040550
3: -0.0019817, 0.0026948, -0.0015198, 0.0026061, -0.0034930, 0.0029112
4: -0.0062076, -0.0018939, -0.0057815, -0.0019757, -0.0042319, 0.0038725
5: 0.0059600, 0.0106281, 0.0064212, 0.0105395, -0.0034727, 0.0028929
6: 0.0069738, 0.0106837, 0.0075960, 0.0105097, -0.0035359, 0.0030878
7: -0.0214718, -0.0113381, -0.0212796, -0.0123392, -0.0052825, 0.0059702
8: 0.9622716, 0.9913060, 0.9628222, 0.9884378, -0.0192536, 0.0229184
9: 0.0002083, 0.0087416, 0.0010513, 0.0085797, -0.0054920, 0.0046439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A2_A2_A1_B1

### Relational analysis result of IS_A2_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118923, upper bound: 0.0110459
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A2_A2_A1_B2

### Relational analysis result of IS_A2_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118923, upper bound: 0.0110459
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0004766, 0.0012603, -0.0004616, 0.0009839, -0.0011086, 0.0013195
1: -0.0009108, 0.0032530, -0.0008406, 0.0028294, -0.0029567, 0.0034407
2: 0.0114682, 0.0177040, 0.0121026, 0.0175989, -0.0048311, 0.0040967
3: -0.0020033, 0.0026858, -0.0015263, 0.0026067, -0.0034846, 0.0029400
4: -0.0062275, -0.0019023, -0.0057875, -0.0019752, -0.0042523, 0.0038852
5: 0.0059385, 0.0106190, 0.0064147, 0.0105402, -0.0034640, 0.0029217
6: 0.0069448, 0.0106918, 0.0075872, 0.0105122, -0.0035673, 0.0031046
7: -0.0214522, -0.0112915, -0.0212810, -0.0123251, -0.0053209, 0.0059277
8: 0.9623277, 0.9914396, 0.9628184, 0.9884782, -0.0194558, 0.0228882
9: 0.0001690, 0.0087251, 0.0010394, 0.0085809, -0.0054558, 0.0046900

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A2_A2_A2_B1

### Relational analysis result of IS_A2_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118975, upper bound: 0.0110318
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_A2_A2_B2

### Relational analysis result of IS_A2_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118975, upper bound: 0.0110318
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004870, 0.0012076, -0.0004879, 0.0012420, -0.0012876, 0.0012540
1: -0.0009594, 0.0031724, -0.0009640, 0.0032251, -0.0032063, 0.0032140
2: 0.0115890, 0.0177767, 0.0115101, 0.0177837, -0.0044235, 0.0044256
3: -0.0019124, 0.0027405, -0.0019718, 0.0027457, -0.0031493, 0.0031565
4: -0.0061437, -0.0018518, -0.0061984, -0.0018470, -0.0042736, 0.0042366
5: 0.0060292, 0.0106736, 0.0059700, 0.0106789, -0.0031266, 0.0031343
6: 0.0070672, 0.0106576, 0.0069872, 0.0106800, -0.0036128, 0.0036704
7: -0.0215707, -0.0114883, -0.0215821, -0.0113597, -0.0053373, 0.0052364
8: 0.9619883, 0.9908757, 0.9619557, 0.9912441, -0.0210508, 0.0210530
9: 0.0003348, 0.0088248, 0.0002265, 0.0088344, -0.0047480, 0.0048064

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A1_A1_B1_B1

### Relational analysis result of IS_A2_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119905, upper bound: 0.0111584
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_A1_B1_B2

### Relational analysis result of IS_A2_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119905, upper bound: 0.0111584
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004870, 0.0012076, -0.0004662, 0.0013474, -0.0013991, 0.0012497
1: -0.0009594, 0.0031724, -0.0008623, 0.0033866, -0.0034392, 0.0032890
2: 0.0115890, 0.0177767, 0.0112682, 0.0176314, -0.0044999, 0.0047743
3: -0.0019124, 0.0027405, -0.0021537, 0.0026312, -0.0031937, 0.0034187
4: -0.0061437, -0.0018518, -0.0063662, -0.0019526, -0.0041910, 0.0044784
5: 0.0060292, 0.0106736, 0.0057884, 0.0105645, -0.0031698, 0.0033961
6: 0.0070672, 0.0106576, 0.0067423, 0.0107485, -0.0036813, 0.0039153
7: -0.0215707, -0.0114883, -0.0213339, -0.0109656, -0.0059055, 0.0052411
8: 0.9619883, 0.9908757, 0.9626669, 0.9923733, -0.0226787, 0.0214409
9: 0.0003348, 0.0088248, -0.0001054, 0.0086254, -0.0048028, 0.0052849

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A1_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119905, upper bound: 0.0111585
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119905, upper bound: 0.0111585
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004653, 0.0013117, -0.0004873, 0.0012072, -0.0012499, 0.0013705
1: -0.0008580, 0.0033318, -0.0009611, 0.0031717, -0.0033055, 0.0033936
2: 0.0113502, 0.0176249, 0.0115901, 0.0177793, -0.0047075, 0.0045310
3: -0.0020920, 0.0026263, -0.0019116, 0.0027424, -0.0033675, 0.0032183
4: -0.0063093, -0.0019571, -0.0061429, -0.0018500, -0.0044277, 0.0041858
5: 0.0058500, 0.0105597, 0.0060300, 0.0106756, -0.0033449, 0.0031945
6: 0.0068253, 0.0107253, 0.0070682, 0.0106573, -0.0038320, 0.0036571
7: -0.0213233, -0.0110992, -0.0215749, -0.0114900, -0.0052131, 0.0058545
8: 0.9626971, 0.9919905, 0.9619762, 0.9908708, -0.0215862, 0.0223667
9: 0.0000071, 0.0086165, 0.0003362, 0.0088284, -0.0052325, 0.0047851

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A2_B2_A1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118996, upper bound: 0.0110574
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119031, upper bound: 0.0110359
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004653, 0.0013117, -0.0004988, 0.0011747, -0.0012200, 0.0013832
1: -0.0008580, 0.0033318, -0.0010147, 0.0031218, -0.0032606, 0.0033888
2: 0.0113502, 0.0176249, 0.0116647, 0.0178595, -0.0047158, 0.0044638
3: -0.0020920, 0.0026263, -0.0018555, 0.0028027, -0.0033823, 0.0031678
4: -0.0063093, -0.0019571, -0.0060912, -0.0017944, -0.0043833, 0.0041340
5: 0.0058500, 0.0105597, 0.0060860, 0.0107358, -0.0033608, 0.0031441
6: 0.0068253, 0.0107253, 0.0071438, 0.0106362, -0.0038109, 0.0035815
7: -0.0213233, -0.0110992, -0.0217056, -0.0116116, -0.0051037, 0.0060059
8: 0.9626971, 0.9919905, 0.9616016, 0.9905225, -0.0212727, 0.0223879
9: 0.0000071, 0.0086165, 0.0004386, 0.0089385, -0.0053419, 0.0046930

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A2_B2_A1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118996, upper bound: 0.0110574
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119031, upper bound: 0.0110359
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004983, 0.0011751, -0.0004875, 0.0012220, -0.0012831, 0.0012298
1: -0.0010123, 0.0031225, -0.0009620, 0.0031945, -0.0031713, 0.0031646
2: 0.0116637, 0.0178561, 0.0115559, 0.0177807, -0.0043564, 0.0043935
3: -0.0018563, 0.0028001, -0.0019373, 0.0027434, -0.0031023, 0.0031407
4: -0.0060918, -0.0017968, -0.0061666, -0.0018491, -0.0042052, 0.0041608
5: 0.0060853, 0.0107332, 0.0060044, 0.0106766, -0.0030801, 0.0031197
6: 0.0071428, 0.0106365, 0.0070336, 0.0106670, -0.0035242, 0.0036028
7: -0.0217000, -0.0116100, -0.0215771, -0.0114343, -0.0054424, 0.0051872
8: 0.9616177, 0.9905269, 0.9619700, 0.9910303, -0.0208814, 0.0207327
9: 0.0004373, 0.0089337, 0.0002893, 0.0088302, -0.0046910, 0.0048635

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A2_B2_A2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118806, upper bound: 0.0110315
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118857, upper bound: 0.0110433
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004983, 0.0011751, -0.0004658, 0.0013289, -0.0013950, 0.0012256
1: -0.0010123, 0.0031225, -0.0008605, 0.0033583, -0.0034024, 0.0032397
2: 0.0116637, 0.0178561, 0.0113106, 0.0176286, -0.0044355, 0.0047396
3: -0.0018563, 0.0028001, -0.0021218, 0.0026291, -0.0031487, 0.0034010
4: -0.0060918, -0.0017968, -0.0063368, -0.0019546, -0.0041373, 0.0044009
5: 0.0060853, 0.0107332, 0.0058202, 0.0105625, -0.0031253, 0.0033795
6: 0.0071428, 0.0106365, 0.0067852, 0.0107365, -0.0035937, 0.0038513
7: -0.0217000, -0.0116100, -0.0213294, -0.0110346, -0.0060063, 0.0051916
8: 0.9616177, 0.9905269, 0.9626797, 0.9921757, -0.0224972, 0.0211305
9: 0.0004373, 0.0089337, -0.0000473, 0.0086216, -0.0047468, 0.0053384

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A2_B2_A2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118806, upper bound: 0.0110315
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118857, upper bound: 0.0110433
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0004783, 0.0012478, -0.0004876, 0.0012115, -0.0012658, 0.0013100
1: -0.0009188, 0.0032339, -0.0009622, 0.0031784, -0.0032506, 0.0033267
2: 0.0114969, 0.0177161, 0.0115801, 0.0177810, -0.0046033, 0.0044756
3: -0.0019817, 0.0026948, -0.0019192, 0.0027437, -0.0032883, 0.0031884
4: -0.0062076, -0.0018939, -0.0061499, -0.0018489, -0.0043587, 0.0042560
5: 0.0059600, 0.0106281, 0.0060225, 0.0106768, -0.0032658, 0.0031658
6: 0.0069738, 0.0106837, 0.0070581, 0.0106602, -0.0036864, 0.0036257
7: -0.0214718, -0.0113381, -0.0215776, -0.0114737, -0.0053293, 0.0056740
8: 0.9622716, 0.9913060, 0.9619684, 0.9909176, -0.0212944, 0.0218833
9: 0.0002083, 0.0087416, 0.0003225, 0.0088307, -0.0050830, 0.0048322

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A2_A2_A1_B1

### Relational analysis result of IS_A2_B2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118923, upper bound: 0.0110588
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_A2_A1_B2

### Relational analysis result of IS_A2_B2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118923, upper bound: 0.0110588
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0004766, 0.0012603, -0.0004877, 0.0012160, -0.0012734, 0.0013220
1: -0.0009108, 0.0032530, -0.0009631, 0.0031852, -0.0032837, 0.0033413
2: 0.0114682, 0.0177040, 0.0115699, 0.0177823, -0.0046224, 0.0045258
3: -0.0020033, 0.0026858, -0.0019269, 0.0027447, -0.0033033, 0.0032268
4: -0.0062275, -0.0019023, -0.0061570, -0.0018479, -0.0043795, 0.0042547
5: 0.0059385, 0.0106190, 0.0060148, 0.0106778, -0.0032810, 0.0032040
6: 0.0069448, 0.0106918, 0.0070477, 0.0106631, -0.0037182, 0.0036441
7: -0.0214522, -0.0112915, -0.0215798, -0.0114570, -0.0053880, 0.0056922
8: 0.9623277, 0.9914396, 0.9619621, 0.9909652, -0.0215275, 0.0219758
9: 0.0001690, 0.0087251, 0.0003085, 0.0088325, -0.0050961, 0.0048977

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A2_A2_A2_B1

### Relational analysis result of IS_A2_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118975, upper bound: 0.0110433
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_A2_A2_B2

### Relational analysis result of IS_A2_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118975, upper bound: 0.0110433
time: 0.76 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.88 seconds
IS_A1_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0114056, upper bound: 0.0114012
IS_A1_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0114056, upper bound: 0.0114012
IS_A1_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0114056, upper bound: 0.0114012
IS_A1_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0114056, upper bound: 0.0114012
IS_A1_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0112659, upper bound: 0.0112853
IS_A1_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0112781, upper bound: 0.0112842
IS_A1_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0112659, upper bound: 0.0112853
IS_A1_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0112781, upper bound: 0.0112842
IS_A1_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0112949, upper bound: 0.0112711
IS_A1_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0112956, upper bound: 0.0112860
IS_A1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0112949, upper bound: 0.0112711
IS_A1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0112956, upper bound: 0.0112860
IS_A1_B1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0112622, upper bound: 0.0112740
IS_A1_B1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0112622, upper bound: 0.0112740
IS_A1_B1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0112736, upper bound: 0.0112736
IS_A1_B1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0112736, upper bound: 0.0112736
IS_A1_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0111856, upper bound: 0.0120231
IS_A1_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0111856, upper bound: 0.0120282
IS_A1_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0111856, upper bound: 0.0120239
IS_A1_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0111856, upper bound: 0.0120282
IS_A1_B2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0110084, upper bound: 0.0118954
IS_A1_B2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0110280, upper bound: 0.0119031
IS_A1_B2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0110084, upper bound: 0.0118954
IS_A1_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0110280, upper bound: 0.0119031
IS_A1_B2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0110190, upper bound: 0.0118806
IS_A1_B2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0110321, upper bound: 0.0118857
IS_A1_B2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0110190, upper bound: 0.0118806
IS_A1_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0110321, upper bound: 0.0118857
IS_A1_B2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0110459, upper bound: 0.0118923
IS_A1_B2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0110459, upper bound: 0.0118761
IS_A1_B2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0110318, upper bound: 0.0118975
IS_A1_B2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0110318, upper bound: 0.0118845
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0120230, upper bound: 0.0111856
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0120230, upper bound: 0.0111856
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0120230, upper bound: 0.0111856
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0120230, upper bound: 0.0111856
IS_A2_B1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0118954, upper bound: 0.0110084
IS_A2_B1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0119031, upper bound: 0.0110280
IS_A2_B1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0118954, upper bound: 0.0110084
IS_A2_B1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0119031, upper bound: 0.0110280
IS_A2_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0118806, upper bound: 0.0110190
IS_A2_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0118857, upper bound: 0.0110321
IS_A2_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0118806, upper bound: 0.0110190
IS_A2_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0118857, upper bound: 0.0110321
IS_A2_B1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0118923, upper bound: 0.0110459
IS_A2_B1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0118923, upper bound: 0.0110459
IS_A2_B1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0118975, upper bound: 0.0110318
IS_A2_B1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0118975, upper bound: 0.0110318
IS_A2_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0119905, upper bound: 0.0111584
IS_A2_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0119905, upper bound: 0.0111584
IS_A2_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0119905, upper bound: 0.0111585
IS_A2_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0119905, upper bound: 0.0111585
IS_A2_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0118996, upper bound: 0.0110574
IS_A2_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0119031, upper bound: 0.0110359
IS_A2_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0118996, upper bound: 0.0110574
IS_A2_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0119031, upper bound: 0.0110359
IS_A2_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0118806, upper bound: 0.0110315
IS_A2_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0118857, upper bound: 0.0110433
IS_A2_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0118806, upper bound: 0.0110315
IS_A2_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0118857, upper bound: 0.0110433
IS_A2_B2_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0118923, upper bound: 0.0110588
IS_A2_B2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0118923, upper bound: 0.0110588
IS_A2_B2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0118975, upper bound: 0.0110433
IS_A2_B2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 8, lower bound: -0.0118975, upper bound: 0.0110433

## BFS IS instance: IS_A1_B1_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0004619, 0.0009759, -0.0004619, 0.0009759, -0.0010003, 0.0010003
1: -0.0008421, 0.0028172, -0.0008421, 0.0028172, -0.0027705, 0.0027705
2: 0.0121210, 0.0176011, 0.0121210, 0.0176011, -0.0038285, 0.0038285
3: -0.0015124, 0.0026084, -0.0015124, 0.0026084, -0.0027331, 0.0027331
4: -0.0057747, -0.0019736, -0.0057747, -0.0019736, -0.0036194, 0.0036194
5: 0.0064285, 0.0105418, 0.0064285, 0.0105418, -0.0027141, 0.0027141
6: 0.0076059, 0.0105070, 0.0076059, 0.0105070, -0.0029011, 0.0029011
7: -0.0212845, -0.0123551, -0.0212845, -0.0123551, -0.0042881, 0.0042881
8: 0.9628081, 0.9883922, 0.9628081, 0.9883922, -0.0182056, 0.0182056
9: 0.0010647, 0.0085839, 0.0010647, 0.0085839, -0.0040904, 0.0040904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A1_B1_A1_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114753, upper bound: 0.0115048
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114857, upper bound: 0.0114855
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0004619, 0.0009759, -0.0004722, 0.0009407, -0.0009683, 0.0010224
1: -0.0008421, 0.0028172, -0.0008903, 0.0027633, -0.0027628, 0.0027697
2: 0.0121210, 0.0176011, 0.0122016, 0.0176734, -0.0038452, 0.0038170
3: -0.0015124, 0.0026084, -0.0014518, 0.0026627, -0.0027532, 0.0027244
4: -0.0057747, -0.0019736, -0.0057188, -0.0019235, -0.0035849, 0.0036114
5: 0.0064285, 0.0105418, 0.0064890, 0.0105960, -0.0027349, 0.0027054
6: 0.0076059, 0.0105070, 0.0076875, 0.0104841, -0.0028782, 0.0028194
7: -0.0212845, -0.0123551, -0.0214023, -0.0124864, -0.0042693, 0.0044991
8: 0.9628081, 0.9883922, 0.9624710, 0.9880158, -0.0181516, 0.0182647
9: 0.0010647, 0.0085839, 0.0011753, 0.0086830, -0.0042087, 0.0040745

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A1_B1_A1_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114753, upper bound: 0.0115048
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114857, upper bound: 0.0114855
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0004619, 0.0009759, -0.0004366, 0.0010694, -0.0011019, 0.0010014
1: -0.0008421, 0.0028172, -0.0007235, 0.0029606, -0.0030438, 0.0028146
2: 0.0121210, 0.0176011, 0.0119062, 0.0174235, -0.0038695, 0.0042378
3: -0.0015124, 0.0026084, -0.0016739, 0.0024749, -0.0027542, 0.0030408
4: -0.0057747, -0.0019736, -0.0059237, -0.0020968, -0.0036779, 0.0039033
5: 0.0064285, 0.0105418, 0.0062673, 0.0104085, -0.0027344, 0.0030213
6: 0.0076059, 0.0105070, 0.0073884, 0.0105678, -0.0029619, 0.0031186
7: -0.0212845, -0.0123551, -0.0209952, -0.0120051, -0.0049550, 0.0043266
8: 0.9628081, 0.9883922, 0.9636371, 0.9893949, -0.0201161, 0.0184204
9: 0.0010647, 0.0085839, 0.0007700, 0.0083402, -0.0041143, 0.0046519

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112938, upper bound: 0.0112716
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112952, upper bound: 0.0112906
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0004619, 0.0009759, -0.0004476, 0.0010348, -0.0010710, 0.0010181
1: -0.0008421, 0.0028172, -0.0007750, 0.0029076, -0.0030106, 0.0028111
2: 0.0121210, 0.0176011, 0.0119856, 0.0175007, -0.0038801, 0.0041881
3: -0.0015124, 0.0026084, -0.0016142, 0.0025329, -0.0027692, 0.0030034
4: -0.0057747, -0.0019736, -0.0058686, -0.0020433, -0.0036933, 0.0038688
5: 0.0064285, 0.0105418, 0.0063269, 0.0104664, -0.0027502, 0.0029840
6: 0.0076059, 0.0105070, 0.0074688, 0.0105453, -0.0029394, 0.0030382
7: -0.0212845, -0.0123551, -0.0211209, -0.0121345, -0.0048739, 0.0044856
8: 0.9628081, 0.9883922, 0.9632769, 0.9890243, -0.0198840, 0.0184552
9: 0.0010647, 0.0085839, 0.0008789, 0.0084461, -0.0042010, 0.0045837

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112938, upper bound: 0.0112716
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112952, upper bound: 0.0112906
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004411, 0.0010393, -0.0004611, 0.0009658, -0.0009868, 0.0010770
1: -0.0007448, 0.0029144, -0.0008383, 0.0028017, -0.0028502, 0.0029756
2: 0.0119754, 0.0174555, 0.0121442, 0.0175955, -0.0041428, 0.0039214
3: -0.0016219, 0.0024989, -0.0014950, 0.0026042, -0.0029723, 0.0027886
4: -0.0058757, -0.0020747, -0.0057586, -0.0019776, -0.0038242, 0.0036840
5: 0.0063192, 0.0104325, 0.0064459, 0.0105376, -0.0029532, 0.0027683
6: 0.0074584, 0.0105482, 0.0076293, 0.0105004, -0.0030420, 0.0029189
7: -0.0210472, -0.0121178, -0.0212754, -0.0123928, -0.0042352, 0.0048792
8: 0.9634882, 0.9890720, 0.9628345, 0.9882841, -0.0186693, 0.0196651
9: 0.0008649, 0.0083840, 0.0010965, 0.0085761, -0.0045585, 0.0041141

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113562, upper bound: 0.0113686
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113562, upper bound: 0.0113686
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004362, 0.0010441, -0.0004612, 0.0009695, -0.0009945, 0.0010778
1: -0.0007217, 0.0029218, -0.0008388, 0.0028074, -0.0028686, 0.0029745
2: 0.0119643, 0.0174207, 0.0121356, 0.0175962, -0.0041378, 0.0039508
3: -0.0016303, 0.0024728, -0.0015014, 0.0026047, -0.0029659, 0.0028121
4: -0.0058834, -0.0020988, -0.0057646, -0.0019770, -0.0038288, 0.0036658
5: 0.0063109, 0.0104064, 0.0064395, 0.0105381, -0.0029464, 0.0027920
6: 0.0074472, 0.0105513, 0.0076207, 0.0105028, -0.0030556, 0.0029307
7: -0.0209906, -0.0120997, -0.0212766, -0.0123789, -0.0043107, 0.0048344
8: 0.9636502, 0.9891238, 0.9628309, 0.9883239, -0.0188031, 0.0196452
9: 0.0008497, 0.0083364, 0.0010847, 0.0085771, -0.0045265, 0.0041693

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113628, upper bound: 0.0113577
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113628, upper bound: 0.0113577
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004411, 0.0010393, -0.0004713, 0.0009305, -0.0009547, 0.0010966
1: -0.0007448, 0.0029144, -0.0008863, 0.0027476, -0.0028408, 0.0029748
2: 0.0119754, 0.0174555, 0.0122251, 0.0176674, -0.0041574, 0.0039073
3: -0.0016219, 0.0024989, -0.0014341, 0.0026582, -0.0029900, 0.0027780
4: -0.0058757, -0.0020747, -0.0057025, -0.0019277, -0.0037885, 0.0036278
5: 0.0063192, 0.0104325, 0.0065066, 0.0105915, -0.0029716, 0.0027577
6: 0.0074584, 0.0105482, 0.0077113, 0.0104775, -0.0030191, 0.0028369
7: -0.0210472, -0.0121178, -0.0213925, -0.0125247, -0.0042123, 0.0050711
8: 0.9634882, 0.9890720, 0.9624989, 0.9879061, -0.0186035, 0.0197190
9: 0.0008649, 0.0083840, 0.0012075, 0.0086747, -0.0046649, 0.0040947

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112659, upper bound: 0.0112829
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112659, upper bound: 0.0112829
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004362, 0.0010441, -0.0004715, 0.0009343, -0.0009625, 0.0010962
1: -0.0007217, 0.0029218, -0.0008868, 0.0027534, -0.0028616, 0.0029716
2: 0.0119643, 0.0174207, 0.0122164, 0.0176681, -0.0041491, 0.0039404
3: -0.0016303, 0.0024728, -0.0014407, 0.0026588, -0.0029816, 0.0028043
4: -0.0058834, -0.0020988, -0.0057085, -0.0019272, -0.0037929, 0.0036097
5: 0.0063109, 0.0104064, 0.0065001, 0.0105921, -0.0029629, 0.0027842
6: 0.0074472, 0.0105513, 0.0077025, 0.0104799, -0.0030328, 0.0028488
7: -0.0209906, -0.0120997, -0.0213937, -0.0125106, -0.0042938, 0.0050132
8: 0.9636502, 0.9891238, 0.9624954, 0.9879467, -0.0187547, 0.0196836
9: 0.0008497, 0.0083364, 0.0011956, 0.0086758, -0.0046383, 0.0041550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112781, upper bound: 0.0112837
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112781, upper bound: 0.0112837
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0004720, 0.0009310, -0.0004669, 0.0009589, -0.0010049, 0.0009631
1: -0.0008894, 0.0027484, -0.0008653, 0.0027912, -0.0027364, 0.0027298
2: 0.0122239, 0.0176720, 0.0121599, 0.0176359, -0.0037698, 0.0037987
3: -0.0014350, 0.0026617, -0.0014832, 0.0026346, -0.0026891, 0.0027198
4: -0.0057033, -0.0019245, -0.0057477, -0.0019495, -0.0035684, 0.0035424
5: 0.0065058, 0.0105950, 0.0064577, 0.0105679, -0.0026702, 0.0027018
6: 0.0077101, 0.0104778, 0.0076453, 0.0104959, -0.0027858, 0.0028325
7: -0.0214001, -0.0125228, -0.0213413, -0.0124185, -0.0044446, 0.0042043
8: 0.9624772, 0.9879116, 0.9626456, 0.9882106, -0.0180442, 0.0179299
9: 0.0012059, 0.0086811, 0.0011181, 0.0086316, -0.0040155, 0.0041577

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115033, upper bound: 0.0114888
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115033, upper bound: 0.0114888
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0004721, 0.0009348, -0.0004619, 0.0009633, -0.0010028, 0.0009708
1: -0.0008899, 0.0027542, -0.0008421, 0.0027979, -0.0027338, 0.0027482
2: 0.0122153, 0.0176727, 0.0121498, 0.0176011, -0.0038002, 0.0037898
3: -0.0014415, 0.0026623, -0.0014908, 0.0026084, -0.0027140, 0.0027108
4: -0.0057093, -0.0019240, -0.0057547, -0.0019737, -0.0035870, 0.0035490
5: 0.0064993, 0.0105956, 0.0064501, 0.0105418, -0.0026952, 0.0026926
6: 0.0077014, 0.0104803, 0.0076350, 0.0104988, -0.0027975, 0.0028452
7: -0.0214013, -0.0125087, -0.0212844, -0.0124020, -0.0043787, 0.0042727
8: 0.9624736, 0.9879520, 0.9628084, 0.9882579, -0.0180066, 0.0180659
9: 0.0011940, 0.0086822, 0.0011042, 0.0085838, -0.0040690, 0.0041232

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114852, upper bound: 0.0114943
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_B2

### Relational analysis result of IS_A1_B1_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114852, upper bound: 0.0114943
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0004720, 0.0009310, -0.0004415, 0.0010525, -0.0011078, 0.0009661
1: -0.0008894, 0.0027484, -0.0007464, 0.0029346, -0.0029958, 0.0027740
2: 0.0122239, 0.0176720, 0.0119451, 0.0174578, -0.0038156, 0.0041872
3: -0.0014350, 0.0026617, -0.0016447, 0.0025006, -0.0027143, 0.0030119
4: -0.0057033, -0.0019245, -0.0058967, -0.0020730, -0.0036303, 0.0038119
5: 0.0065058, 0.0105950, 0.0062965, 0.0104342, -0.0026945, 0.0029933
6: 0.0077101, 0.0104778, 0.0074278, 0.0105568, -0.0028467, 0.0030500
7: -0.0214001, -0.0125228, -0.0210510, -0.0120685, -0.0050776, 0.0042665
8: 0.9624772, 0.9879116, 0.9634772, 0.9892132, -0.0198578, 0.0181644
9: 0.0012059, 0.0086811, 0.0008234, 0.0083872, -0.0040457, 0.0046907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112923, upper bound: 0.0112711
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112923, upper bound: 0.0112711
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0004721, 0.0009348, -0.0004365, 0.0010577, -0.0011067, 0.0009723
1: -0.0008899, 0.0027542, -0.0007231, 0.0029425, -0.0029937, 0.0027929
2: 0.0122153, 0.0176727, 0.0119333, 0.0174229, -0.0038444, 0.0041791
3: -0.0014415, 0.0026623, -0.0016536, 0.0024744, -0.0027372, 0.0030036
4: -0.0057093, -0.0019240, -0.0059049, -0.0020973, -0.0036120, 0.0038190
5: 0.0064993, 0.0105956, 0.0062876, 0.0104081, -0.0027177, 0.0029848
6: 0.0077014, 0.0104803, 0.0074158, 0.0105601, -0.0028588, 0.0030645
7: -0.0214013, -0.0125087, -0.0209942, -0.0120492, -0.0050131, 0.0043144
8: 0.9624736, 0.9879520, 0.9636401, 0.9892687, -0.0198240, 0.0182971
9: 0.0011940, 0.0086822, 0.0008071, 0.0083394, -0.0040946, 0.0046574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112923, upper bound: 0.0112860
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112923, upper bound: 0.0112860
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004518, 0.0010047, -0.0004621, 0.0009805, -0.0010196, 0.0010490
1: -0.0007948, 0.0028614, -0.0008430, 0.0028242, -0.0027833, 0.0029987
2: 0.0120548, 0.0175302, 0.0121104, 0.0176024, -0.0041692, 0.0038463
3: -0.0015622, 0.0025551, -0.0015204, 0.0026094, -0.0029884, 0.0027471
4: -0.0058206, -0.0020228, -0.0057820, -0.0019728, -0.0038479, 0.0036449
5: 0.0063788, 0.0104886, 0.0064206, 0.0105428, -0.0029689, 0.0027285
6: 0.0075388, 0.0105257, 0.0075951, 0.0105100, -0.0029712, 0.0029306
7: -0.0211691, -0.0122471, -0.0212866, -0.0123378, -0.0044912, 0.0048247
8: 0.9631390, 0.9887015, 0.9628021, 0.9884416, -0.0182905, 0.0197960
9: 0.0009738, 0.0084866, 0.0010502, 0.0085856, -0.0045490, 0.0041921

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_A2_A1_B1_B1

### Relational analysis result of IS_A1_B1_A2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112602, upper bound: 0.0112740
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_A2_A1_B1_B2

### Relational analysis result of IS_A1_B1_A2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112602, upper bound: 0.0112740
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004518, 0.0010047, -0.0004367, 0.0010733, -0.0010927, 0.0010174
1: -0.0007948, 0.0028614, -0.0007240, 0.0029665, -0.0029244, 0.0029137
2: 0.0120548, 0.0175302, 0.0118974, 0.0174243, -0.0040082, 0.0040477
3: -0.0015622, 0.0025551, -0.0016806, 0.0024754, -0.0028504, 0.0028912
4: -0.0058206, -0.0020228, -0.0059298, -0.0020963, -0.0037243, 0.0038001
5: 0.0063788, 0.0104886, 0.0062607, 0.0104091, -0.0028297, 0.0028715
6: 0.0075388, 0.0105257, 0.0073794, 0.0105703, -0.0030315, 0.0031463
7: -0.0211691, -0.0122471, -0.0209964, -0.0119908, -0.0046083, 0.0043221
8: 0.9631390, 0.9887015, 0.9636337, 0.9894360, -0.0192427, 0.0190827
9: 0.0009738, 0.0084866, 0.0007579, 0.0083412, -0.0042056, 0.0043736

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_A2_A1_B2_B1

### Relational analysis result of IS_A1_B1_A2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112602, upper bound: 0.0112740
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_A2_A1_B2_B2

### Relational analysis result of IS_A1_B1_A2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112602, upper bound: 0.0112740
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004472, 0.0010101, -0.0004622, 0.0009842, -0.0010264, 0.0010488
1: -0.0007732, 0.0028696, -0.0008435, 0.0028300, -0.0028047, 0.0029959
2: 0.0120425, 0.0174979, 0.0121018, 0.0176032, -0.0041590, 0.0038796
3: -0.0015715, 0.0025308, -0.0015269, 0.0026100, -0.0029779, 0.0027734
4: -0.0058291, -0.0020452, -0.0057880, -0.0019722, -0.0038570, 0.0036664
5: 0.0063696, 0.0104644, 0.0064141, 0.0105434, -0.0029581, 0.0027548
6: 0.0075264, 0.0105292, 0.0075864, 0.0105124, -0.0029860, 0.0029428
7: -0.0211164, -0.0122271, -0.0212880, -0.0123238, -0.0045495, 0.0047763
8: 0.9632899, 0.9887587, 0.9627982, 0.9884818, -0.0184439, 0.0197553
9: 0.0009570, 0.0084423, 0.0010383, 0.0085868, -0.0045178, 0.0042374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_A2_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112716, upper bound: 0.0112736
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_A2_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112716, upper bound: 0.0112736
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004472, 0.0010101, -0.0004368, 0.0010770, -0.0011016, 0.0010182
1: -0.0007732, 0.0028696, -0.0007246, 0.0029721, -0.0029434, 0.0029126
2: 0.0120425, 0.0174979, 0.0118889, 0.0174252, -0.0040033, 0.0040792
3: -0.0015715, 0.0025308, -0.0016869, 0.0024761, -0.0028445, 0.0029158
4: -0.0058291, -0.0020452, -0.0059357, -0.0020957, -0.0037335, 0.0038190
5: 0.0063696, 0.0104644, 0.0062543, 0.0104097, -0.0028233, 0.0028960
6: 0.0075264, 0.0105292, 0.0073709, 0.0105727, -0.0030463, 0.0031583
7: -0.0211164, -0.0122271, -0.0209978, -0.0119769, -0.0046959, 0.0042722
8: 0.9632899, 0.9887587, 0.9636295, 0.9894756, -0.0193859, 0.0190632
9: 0.0009570, 0.0084423, 0.0007463, 0.0083424, -0.0041780, 0.0044218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_A2_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112716, upper bound: 0.0112736
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_A2_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112716, upper bound: 0.0112736
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004619, 0.0009759, -0.0004870, 0.0012076, -0.0012493, 0.0010944
1: -0.0008421, 0.0028172, -0.0009594, 0.0031724, -0.0032591, 0.0028457
2: 0.0121210, 0.0176011, 0.0115890, 0.0177767, -0.0039556, 0.0045602
3: -0.0015124, 0.0026084, -0.0019124, 0.0027405, -0.0028460, 0.0032833
4: -0.0057747, -0.0019736, -0.0061437, -0.0018518, -0.0037647, 0.0041269
5: 0.0064285, 0.0105418, 0.0060292, 0.0106736, -0.0028289, 0.0032633
6: 0.0076059, 0.0105070, 0.0070672, 0.0106576, -0.0030518, 0.0034398
7: -0.0212845, -0.0123551, -0.0215707, -0.0114883, -0.0054803, 0.0052120
8: 0.9628081, 0.9883922, 0.9619883, 0.9908757, -0.0216215, 0.0187679
9: 0.0010647, 0.0085839, 0.0003348, 0.0088248, -0.0045796, 0.0050943

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A1_B2_B1_A1_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112113, upper bound: 0.0120867
time: 0.70 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_A1_A2

### Relational analysis result of IS_A1_B2_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112234, upper bound: 0.0120872
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004722, 0.0009407, -0.0004870, 0.0012076, -0.0012714, 0.0010607
1: -0.0008903, 0.0027633, -0.0009594, 0.0031724, -0.0032583, 0.0028023
2: 0.0122016, 0.0176734, 0.0115890, 0.0177767, -0.0038905, 0.0045769
3: -0.0014518, 0.0026627, -0.0019124, 0.0027405, -0.0027971, 0.0033034
4: -0.0057188, -0.0019235, -0.0061437, -0.0018518, -0.0037195, 0.0040924
5: 0.0064890, 0.0105960, 0.0060292, 0.0106736, -0.0027800, 0.0032841
6: 0.0076875, 0.0104841, 0.0070672, 0.0106576, -0.0029701, 0.0034170
7: -0.0214023, -0.0124864, -0.0215707, -0.0114883, -0.0056913, 0.0051060
8: 0.9624710, 0.9880158, 0.9619883, 0.9908757, -0.0216805, 0.0184641
9: 0.0011753, 0.0086830, 0.0003348, 0.0088248, -0.0044903, 0.0052126

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112531, upper bound: 0.0120935
time: 0.79 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112234, upper bound: 0.0120963
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004619, 0.0009759, -0.0004653, 0.0013117, -0.0013652, 0.0010857
1: -0.0008421, 0.0028172, -0.0008580, 0.0033318, -0.0034883, 0.0028978
2: 0.0121210, 0.0176011, 0.0113502, 0.0176249, -0.0039932, 0.0049035
3: -0.0015124, 0.0026084, -0.0020920, 0.0026263, -0.0028603, 0.0035414
4: -0.0057747, -0.0019736, -0.0063093, -0.0019571, -0.0038175, 0.0043357
5: 0.0064285, 0.0105418, 0.0058500, 0.0105597, -0.0028421, 0.0035210
6: 0.0076059, 0.0105070, 0.0068253, 0.0107253, -0.0031194, 0.0036816
7: -0.0212845, -0.0123551, -0.0213233, -0.0110992, -0.0060397, 0.0051200
8: 0.9628081, 0.9883922, 0.9626971, 0.9919905, -0.0232242, 0.0189822
9: 0.0010647, 0.0085839, 0.0000071, 0.0086165, -0.0045550, 0.0055654

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A1_B2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110740, upper bound: 0.0119194
time: 0.76 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110699, upper bound: 0.0119232
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004722, 0.0009407, -0.0004653, 0.0013117, -0.0013873, 0.0010520
1: -0.0008903, 0.0027633, -0.0008580, 0.0033318, -0.0034875, 0.0028543
2: 0.0122016, 0.0176734, 0.0113502, 0.0176249, -0.0039281, 0.0049202
3: -0.0014518, 0.0026627, -0.0020920, 0.0026263, -0.0028113, 0.0035615
4: -0.0057188, -0.0019235, -0.0063093, -0.0019571, -0.0037616, 0.0043305
5: 0.0064890, 0.0105960, 0.0058500, 0.0105597, -0.0027932, 0.0035418
6: 0.0076875, 0.0104841, 0.0068253, 0.0107253, -0.0030378, 0.0036588
7: -0.0214023, -0.0124864, -0.0213233, -0.0110992, -0.0062507, 0.0050139
8: 0.9624710, 0.9880158, 0.9626971, 0.9919905, -0.0232833, 0.0186784
9: 0.0011753, 0.0086830, 0.0000071, 0.0086165, -0.0044658, 0.0056836

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A1_B2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110740, upper bound: 0.0119227
time: 0.65 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110699, upper bound: 0.0119281
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0004411, 0.0010393, -0.0004870, 0.0011969, -0.0012348, 0.0011681
1: -0.0007448, 0.0029144, -0.0009597, 0.0031559, -0.0033374, 0.0030231
2: 0.0119754, 0.0174555, 0.0116137, 0.0177772, -0.0042303, 0.0046511
3: -0.0016219, 0.0024989, -0.0018939, 0.0027408, -0.0030561, 0.0033373
4: -0.0058757, -0.0020747, -0.0061265, -0.0018515, -0.0039418, 0.0040519
5: 0.0063192, 0.0104325, 0.0060478, 0.0106740, -0.0030391, 0.0033160
6: 0.0074584, 0.0105482, 0.0070922, 0.0106506, -0.0031922, 0.0034560
7: -0.0210472, -0.0121178, -0.0215715, -0.0115285, -0.0054243, 0.0057077
8: 0.9634882, 0.9890720, 0.9619861, 0.9907603, -0.0220760, 0.0200428
9: 0.0008649, 0.0083840, 0.0003687, 0.0088255, -0.0049980, 0.0051153

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_B1_A2_A1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110800, upper bound: 0.0119556
time: 0.75 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110800, upper bound: 0.0119556
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0004362, 0.0010441, -0.0004872, 0.0012014, -0.0012439, 0.0011732
1: -0.0007217, 0.0029218, -0.0009605, 0.0031629, -0.0033575, 0.0030370
2: 0.0119643, 0.0174207, 0.0116033, 0.0177785, -0.0042491, 0.0046830
3: -0.0016303, 0.0024728, -0.0019017, 0.0027417, -0.0030679, 0.0033627
4: -0.0058834, -0.0020988, -0.0061338, -0.0018506, -0.0039549, 0.0040350
5: 0.0063109, 0.0104064, 0.0060399, 0.0106749, -0.0030505, 0.0033416
6: 0.0074472, 0.0105513, 0.0070816, 0.0106536, -0.0032064, 0.0034698
7: -0.0209906, -0.0120997, -0.0215735, -0.0115115, -0.0055039, 0.0057263
8: 0.9636502, 0.9891238, 0.9619802, 0.9908092, -0.0222218, 0.0201359
9: 0.0008497, 0.0083364, 0.0003543, 0.0088272, -0.0050168, 0.0051740

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_B1_A2_A1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110920, upper bound: 0.0119510
time: 0.64 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110920, upper bound: 0.0119510
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0004518, 0.0010047, -0.0004870, 0.0011969, -0.0012548, 0.0011367
1: -0.0007948, 0.0028614, -0.0009597, 0.0031559, -0.0033360, 0.0029661
2: 0.0120548, 0.0175302, 0.0116137, 0.0177772, -0.0041450, 0.0046641
3: -0.0015622, 0.0025551, -0.0018939, 0.0027408, -0.0029919, 0.0033547
4: -0.0058206, -0.0020228, -0.0061265, -0.0018515, -0.0038826, 0.0041037
5: 0.0063788, 0.0104886, 0.0060478, 0.0106740, -0.0029751, 0.0033342
6: 0.0075388, 0.0105257, 0.0070922, 0.0106506, -0.0031118, 0.0034335
7: -0.0211691, -0.0122471, -0.0215715, -0.0115285, -0.0056127, 0.0055687
8: 0.9631390, 0.9887015, 0.9619861, 0.9907603, -0.0221204, 0.0196444
9: 0.0009738, 0.0084866, 0.0003687, 0.0088255, -0.0048809, 0.0052194

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_B1_A2_A2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110084, upper bound: 0.0118820
time: 0.82 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110084, upper bound: 0.0118820
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0004472, 0.0010101, -0.0004872, 0.0012014, -0.0012648, 0.0011418
1: -0.0007732, 0.0028696, -0.0009605, 0.0031629, -0.0033565, 0.0029791
2: 0.0120425, 0.0174979, 0.0116033, 0.0177785, -0.0041624, 0.0046979
3: -0.0015715, 0.0025308, -0.0019017, 0.0027417, -0.0030027, 0.0033810
4: -0.0058291, -0.0020452, -0.0061338, -0.0018506, -0.0038948, 0.0040886
5: 0.0063696, 0.0104644, 0.0060399, 0.0106749, -0.0029855, 0.0033604
6: 0.0075264, 0.0105292, 0.0070816, 0.0106536, -0.0031272, 0.0034476
7: -0.0211164, -0.0122271, -0.0215735, -0.0115115, -0.0057041, 0.0055851
8: 0.9632899, 0.9887587, 0.9619802, 0.9908092, -0.0222744, 0.0197312
9: 0.0009570, 0.0084423, 0.0003543, 0.0088272, -0.0048979, 0.0052708

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_B1_A2_A2_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110280, upper bound: 0.0118867
time: 0.78 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110280, upper bound: 0.0118867
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0004669, 0.0009589, -0.0004980, 0.0011645, -0.0012141, 0.0010866
1: -0.0008653, 0.0027912, -0.0010110, 0.0031062, -0.0032168, 0.0027995
2: 0.0121599, 0.0176359, 0.0116881, 0.0178541, -0.0039074, 0.0044992
3: -0.0014832, 0.0026346, -0.0018380, 0.0027986, -0.0028161, 0.0032375
4: -0.0057477, -0.0019495, -0.0060750, -0.0017982, -0.0036696, 0.0040743
5: 0.0064577, 0.0105679, 0.0061036, 0.0107317, -0.0027997, 0.0032177
6: 0.0076453, 0.0104959, 0.0071675, 0.0106296, -0.0029843, 0.0033284
7: -0.0213413, -0.0124185, -0.0216967, -0.0116497, -0.0053927, 0.0052295
8: 0.9626456, 0.9882106, 0.9616272, 0.9904132, -0.0213351, 0.0185265
9: 0.0011181, 0.0086316, 0.0004707, 0.0089309, -0.0045810, 0.0050163

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B2_B1_A1_A1_A1

### Relational analysis result of IS_A1_B2_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112113, upper bound: 0.0120847
time: 0.80 seconds

## Relational analysis of IS_A1_B2_B2_B1_A1_A1_A2

### Relational analysis result of IS_A1_B2_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112113, upper bound: 0.0120921
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0004619, 0.0009633, -0.0004982, 0.0011695, -0.0012235, 0.0010898
1: -0.0008421, 0.0027979, -0.0010118, 0.0031139, -0.0032370, 0.0028084
2: 0.0121498, 0.0176011, 0.0116766, 0.0178553, -0.0039193, 0.0045322
3: -0.0014908, 0.0026084, -0.0018466, 0.0027995, -0.0028231, 0.0032645
4: -0.0057547, -0.0019737, -0.0060829, -0.0017973, -0.0036796, 0.0040948
5: 0.0064501, 0.0105418, 0.0060949, 0.0107326, -0.0028066, 0.0032447
6: 0.0076350, 0.0104988, 0.0071558, 0.0106328, -0.0029978, 0.0033430
7: -0.0212844, -0.0124020, -0.0216987, -0.0116310, -0.0054656, 0.0052394
8: 0.9628084, 0.9882579, 0.9616215, 0.9904668, -0.0214837, 0.0185841
9: 0.0011042, 0.0085838, 0.0004549, 0.0089326, -0.0045846, 0.0050735

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B2_B1_A1_A2_A1

### Relational analysis result of IS_A1_B2_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112234, upper bound: 0.0120829
time: 0.77 seconds

## Relational analysis of IS_A1_B2_B2_B1_A1_A2_A2

### Relational analysis result of IS_A1_B2_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112234, upper bound: 0.0120963
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0004415, 0.0010525, -0.0004980, 0.0011645, -0.0012172, 0.0011884
1: -0.0007464, 0.0029346, -0.0010110, 0.0031062, -0.0032610, 0.0030357
2: 0.0119451, 0.0174578, 0.0116881, 0.0178541, -0.0042611, 0.0045450
3: -0.0016447, 0.0025006, -0.0018380, 0.0027986, -0.0030822, 0.0032628
4: -0.0058967, -0.0020730, -0.0060750, -0.0017982, -0.0039150, 0.0040019
5: 0.0062965, 0.0104342, 0.0061036, 0.0107317, -0.0030652, 0.0032419
6: 0.0074278, 0.0105568, 0.0071675, 0.0106296, -0.0032018, 0.0033893
7: -0.0210510, -0.0120685, -0.0216967, -0.0116497, -0.0054549, 0.0058059
8: 0.9634772, 0.9892132, 0.9616272, 0.9904132, -0.0215695, 0.0201780
9: 0.0008234, 0.0083872, 0.0004707, 0.0089309, -0.0050664, 0.0050465

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B2_B1_A2_A1_A1

### Relational analysis result of IS_A1_B2_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110052, upper bound: 0.0118740
time: 0.73 seconds

## Relational analysis of IS_A1_B2_B2_B1_A2_A1_A2

### Relational analysis result of IS_A1_B2_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110052, upper bound: 0.0118806
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0004365, 0.0010577, -0.0004982, 0.0011695, -0.0012249, 0.0011934
1: -0.0007231, 0.0029425, -0.0010118, 0.0031139, -0.0032818, 0.0030470
2: 0.0119333, 0.0174229, 0.0116766, 0.0178553, -0.0042767, 0.0045764
3: -0.0016536, 0.0024744, -0.0018466, 0.0027995, -0.0030918, 0.0032877
4: -0.0059049, -0.0020973, -0.0060829, -0.0017973, -0.0039275, 0.0039857
5: 0.0062876, 0.0104081, 0.0060949, 0.0107326, -0.0030749, 0.0032672
6: 0.0074158, 0.0105601, 0.0071558, 0.0106328, -0.0032171, 0.0034043
7: -0.0209942, -0.0120492, -0.0216987, -0.0116310, -0.0055073, 0.0058217
8: 0.9636401, 0.9892687, 0.9616215, 0.9904668, -0.0217148, 0.0202524
9: 0.0008071, 0.0083394, 0.0004549, 0.0089326, -0.0050750, 0.0050990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B2_B1_A2_A2_A1

### Relational analysis result of IS_A1_B2_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110232, upper bound: 0.0118770
time: 0.77 seconds

## Relational analysis of IS_A1_B2_B2_B1_A2_A2_A2

### Relational analysis result of IS_A1_B2_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110232, upper bound: 0.0118857
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004621, 0.0009805, -0.0004783, 0.0012478, -0.0013076, 0.0010966
1: -0.0008430, 0.0028242, -0.0009188, 0.0032339, -0.0034407, 0.0028541
2: 0.0121104, 0.0176024, 0.0114969, 0.0177161, -0.0039618, 0.0048311
3: -0.0015204, 0.0026094, -0.0019817, 0.0026948, -0.0028516, 0.0034862
4: -0.0057820, -0.0019728, -0.0062076, -0.0018939, -0.0037901, 0.0042348
5: 0.0064206, 0.0105428, 0.0059600, 0.0106281, -0.0028346, 0.0034657
6: 0.0075951, 0.0105100, 0.0069738, 0.0106837, -0.0030886, 0.0035361
7: -0.0212866, -0.0123378, -0.0214718, -0.0113381, -0.0059033, 0.0052048
8: 0.9628021, 0.9884416, 0.9622716, 0.9913060, -0.0228863, 0.0188002
9: 0.0010502, 0.0085856, 0.0002083, 0.0087416, -0.0046012, 0.0054572

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B2_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110385, upper bound: 0.0118872
time: 0.72 seconds

## Relational analysis of IS_A1_B2_B2_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110385, upper bound: 0.0118923
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004367, 0.0010733, -0.0004783, 0.0012478, -0.0012755, 0.0011809
1: -0.0007240, 0.0029665, -0.0009188, 0.0032339, -0.0033916, 0.0030010
2: 0.0118974, 0.0174243, 0.0114969, 0.0177161, -0.0041679, 0.0047239
3: -0.0016806, 0.0024754, -0.0019817, 0.0026948, -0.0029961, 0.0033886
4: -0.0059298, -0.0020963, -0.0062076, -0.0018939, -0.0039508, 0.0041113
5: 0.0062607, 0.0104091, 0.0059600, 0.0106281, -0.0029777, 0.0033669
6: 0.0073794, 0.0105703, 0.0069738, 0.0106837, -0.0033043, 0.0035965
7: -0.0209964, -0.0119908, -0.0214718, -0.0113381, -0.0054883, 0.0054665
8: 0.9636337, 0.9894360, 0.9622716, 0.9913060, -0.0224240, 0.0197809
9: 0.0007579, 0.0083412, 0.0002083, 0.0087416, -0.0047989, 0.0051876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B2_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110385, upper bound: 0.0118679
time: 0.72 seconds

## Relational analysis of IS_A1_B2_B2_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110385, upper bound: 0.0118761
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004622, 0.0009842, -0.0004766, 0.0012603, -0.0013123, 0.0011012
1: -0.0008435, 0.0028300, -0.0009108, 0.0032530, -0.0034378, 0.0028839
2: 0.0121018, 0.0176032, 0.0114682, 0.0177040, -0.0039975, 0.0048207
3: -0.0015269, 0.0026100, -0.0020033, 0.0026858, -0.0028782, 0.0034754
4: -0.0057880, -0.0019722, -0.0062275, -0.0019023, -0.0038312, 0.0042553
5: 0.0064141, 0.0105434, 0.0059385, 0.0106190, -0.0028613, 0.0034548
6: 0.0075864, 0.0105124, 0.0069448, 0.0106918, -0.0031054, 0.0035676
7: -0.0212880, -0.0123238, -0.0214522, -0.0112915, -0.0058545, 0.0052346
8: 0.9627982, 0.9884818, 0.9623277, 0.9914396, -0.0228445, 0.0189718
9: 0.0010383, 0.0085868, 0.0001690, 0.0087251, -0.0046358, 0.0054257

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B2_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110232, upper bound: 0.0118935
time: 0.76 seconds

## Relational analysis of IS_A1_B2_B2_B2_B2_A1_A2

### Relational analysis result of IS_A1_B2_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110232, upper bound: 0.0118975
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004368, 0.0010770, -0.0004766, 0.0012603, -0.0012806, 0.0011866
1: -0.0007246, 0.0029721, -0.0009108, 0.0032530, -0.0033902, 0.0030325
2: 0.0118889, 0.0174252, 0.0114682, 0.0177040, -0.0042102, 0.0047186
3: -0.0016869, 0.0024761, -0.0020033, 0.0026858, -0.0030254, 0.0033823
4: -0.0059357, -0.0020957, -0.0062275, -0.0019023, -0.0039929, 0.0041318
5: 0.0062543, 0.0104097, 0.0059385, 0.0106190, -0.0030069, 0.0033602
6: 0.0073709, 0.0105727, 0.0069448, 0.0106918, -0.0033210, 0.0036279
7: -0.0209978, -0.0119769, -0.0214522, -0.0112915, -0.0054377, 0.0055059
8: 0.9636295, 0.9894756, 0.9623277, 0.9914396, -0.0224026, 0.0199858
9: 0.0007463, 0.0083424, 0.0001690, 0.0087251, -0.0048457, 0.0051595

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B2_B2_B2_A2_A1

### Relational analysis result of IS_A1_B2_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110232, upper bound: 0.0118770
time: 0.74 seconds

## Relational analysis of IS_A1_B2_B2_B2_B2_A2_A2

### Relational analysis result of IS_A1_B2_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110232, upper bound: 0.0118845
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004870, 0.0012076, -0.0004619, 0.0009759, -0.0010944, 0.0012493
1: -0.0009594, 0.0031724, -0.0008421, 0.0028172, -0.0028457, 0.0032591
2: 0.0115890, 0.0177767, 0.0121210, 0.0176011, -0.0045602, 0.0039556
3: -0.0019124, 0.0027405, -0.0015124, 0.0026084, -0.0032833, 0.0028460
4: -0.0061437, -0.0018518, -0.0057747, -0.0019736, -0.0041269, 0.0037647
5: 0.0060292, 0.0106736, 0.0064285, 0.0105418, -0.0032633, 0.0028289
6: 0.0070672, 0.0106576, 0.0076059, 0.0105070, -0.0034398, 0.0030518
7: -0.0215707, -0.0114883, -0.0212845, -0.0123551, -0.0052120, 0.0054803
8: 0.9619883, 0.9908757, 0.9628081, 0.9883922, -0.0187679, 0.0216215
9: 0.0003348, 0.0088248, 0.0010647, 0.0085839, -0.0050943, 0.0045796

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120867, upper bound: 0.0112113
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120872, upper bound: 0.0112234
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004870, 0.0012076, -0.0004722, 0.0009407, -0.0010607, 0.0012714
1: -0.0009594, 0.0031724, -0.0008903, 0.0027633, -0.0028023, 0.0032583
2: 0.0115890, 0.0177767, 0.0122016, 0.0176734, -0.0045769, 0.0038905
3: -0.0019124, 0.0027405, -0.0014518, 0.0026627, -0.0033034, 0.0027971
4: -0.0061437, -0.0018518, -0.0057188, -0.0019235, -0.0040924, 0.0037195
5: 0.0060292, 0.0106736, 0.0064890, 0.0105960, -0.0032841, 0.0027800
6: 0.0070672, 0.0106576, 0.0076875, 0.0104841, -0.0034170, 0.0029701
7: -0.0215707, -0.0114883, -0.0214023, -0.0124864, -0.0051060, 0.0056913
8: 0.9619883, 0.9908757, 0.9624710, 0.9880158, -0.0184641, 0.0216805
9: 0.0003348, 0.0088248, 0.0011753, 0.0086830, -0.0052126, 0.0044903

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120852, upper bound: 0.0112531
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120872, upper bound: 0.0112234
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004653, 0.0013117, -0.0004619, 0.0009759, -0.0010857, 0.0013652
1: -0.0008580, 0.0033318, -0.0008421, 0.0028172, -0.0028978, 0.0034883
2: 0.0113502, 0.0176249, 0.0121210, 0.0176011, -0.0049035, 0.0039932
3: -0.0020920, 0.0026263, -0.0015124, 0.0026084, -0.0035414, 0.0028603
4: -0.0063093, -0.0019571, -0.0057747, -0.0019736, -0.0043357, 0.0038175
5: 0.0058500, 0.0105597, 0.0064285, 0.0105418, -0.0035210, 0.0028421
6: 0.0068253, 0.0107253, 0.0076059, 0.0105070, -0.0036816, 0.0031194
7: -0.0213233, -0.0110992, -0.0212845, -0.0123551, -0.0051200, 0.0060397
8: 0.9626971, 0.9919905, 0.9628081, 0.9883922, -0.0189822, 0.0232242
9: 0.0000071, 0.0086165, 0.0010647, 0.0085839, -0.0055654, 0.0045550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119194, upper bound: 0.0110740
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119232, upper bound: 0.0110699
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004653, 0.0013117, -0.0004722, 0.0009407, -0.0010520, 0.0013873
1: -0.0008580, 0.0033318, -0.0008903, 0.0027633, -0.0028543, 0.0034875
2: 0.0113502, 0.0176249, 0.0122016, 0.0176734, -0.0049202, 0.0039281
3: -0.0020920, 0.0026263, -0.0014518, 0.0026627, -0.0035615, 0.0028113
4: -0.0063093, -0.0019571, -0.0057188, -0.0019235, -0.0043305, 0.0037616
5: 0.0058500, 0.0105597, 0.0064890, 0.0105960, -0.0035418, 0.0027932
6: 0.0068253, 0.0107253, 0.0076875, 0.0104841, -0.0036588, 0.0030378
7: -0.0213233, -0.0110992, -0.0214023, -0.0124864, -0.0050139, 0.0062507
8: 0.9626971, 0.9919905, 0.9624710, 0.9880158, -0.0186784, 0.0232833
9: 0.0000071, 0.0086165, 0.0011753, 0.0086830, -0.0056836, 0.0044658

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119194, upper bound: 0.0110740
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119232, upper bound: 0.0110699
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0004870, 0.0011969, -0.0004411, 0.0010393, -0.0011681, 0.0012348
1: -0.0009597, 0.0031559, -0.0007448, 0.0029144, -0.0030231, 0.0033374
2: 0.0116137, 0.0177772, 0.0119754, 0.0174555, -0.0046511, 0.0042303
3: -0.0018939, 0.0027408, -0.0016219, 0.0024989, -0.0033373, 0.0030561
4: -0.0061265, -0.0018515, -0.0058757, -0.0020747, -0.0040519, 0.0039418
5: 0.0060478, 0.0106740, 0.0063192, 0.0104325, -0.0033160, 0.0030391
6: 0.0070922, 0.0106506, 0.0074584, 0.0105482, -0.0034560, 0.0031922
7: -0.0215715, -0.0115285, -0.0210472, -0.0121178, -0.0057077, 0.0054243
8: 0.9619861, 0.9907603, 0.9634882, 0.9890720, -0.0200428, 0.0220760
9: 0.0003687, 0.0088255, 0.0008649, 0.0083840, -0.0051153, 0.0049980

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1_B2_B1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119556, upper bound: 0.0110800
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_B2_B1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119556, upper bound: 0.0110796
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0004872, 0.0012014, -0.0004362, 0.0010441, -0.0011732, 0.0012439
1: -0.0009605, 0.0031629, -0.0007217, 0.0029218, -0.0030370, 0.0033575
2: 0.0116033, 0.0177785, 0.0119643, 0.0174207, -0.0046830, 0.0042491
3: -0.0019017, 0.0027417, -0.0016303, 0.0024728, -0.0033627, 0.0030679
4: -0.0061338, -0.0018506, -0.0058834, -0.0020988, -0.0040350, 0.0039549
5: 0.0060399, 0.0106749, 0.0063109, 0.0104064, -0.0033416, 0.0030505
6: 0.0070816, 0.0106536, 0.0074472, 0.0105513, -0.0034698, 0.0032064
7: -0.0215735, -0.0115115, -0.0209906, -0.0120997, -0.0057263, 0.0055039
8: 0.9619802, 0.9908092, 0.9636502, 0.9891238, -0.0201359, 0.0222218
9: 0.0003543, 0.0088272, 0.0008497, 0.0083364, -0.0051740, 0.0050168

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1_B2_B1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119510, upper bound: 0.0110920
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B2_B1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119510, upper bound: 0.0110920
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0004870, 0.0011969, -0.0004518, 0.0010047, -0.0011367, 0.0012548
1: -0.0009597, 0.0031559, -0.0007948, 0.0028614, -0.0029661, 0.0033360
2: 0.0116137, 0.0177772, 0.0120548, 0.0175302, -0.0046641, 0.0041450
3: -0.0018939, 0.0027408, -0.0015622, 0.0025551, -0.0033547, 0.0029919
4: -0.0061265, -0.0018515, -0.0058206, -0.0020228, -0.0041037, 0.0038826
5: 0.0060478, 0.0106740, 0.0063788, 0.0104886, -0.0033342, 0.0029751
6: 0.0070922, 0.0106506, 0.0075388, 0.0105257, -0.0034335, 0.0031118
7: -0.0215715, -0.0115285, -0.0211691, -0.0122471, -0.0055687, 0.0056127
8: 0.9619861, 0.9907603, 0.9631390, 0.9887015, -0.0196444, 0.0221204
9: 0.0003687, 0.0088255, 0.0009738, 0.0084866, -0.0052194, 0.0048809

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1_B2_B2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118820, upper bound: 0.0110084
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B2_B2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118820, upper bound: 0.0110084
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0004872, 0.0012014, -0.0004472, 0.0010101, -0.0011418, 0.0012648
1: -0.0009605, 0.0031629, -0.0007732, 0.0028696, -0.0029791, 0.0033565
2: 0.0116033, 0.0177785, 0.0120425, 0.0174979, -0.0046979, 0.0041624
3: -0.0019017, 0.0027417, -0.0015715, 0.0025308, -0.0033810, 0.0030027
4: -0.0061338, -0.0018506, -0.0058291, -0.0020452, -0.0040886, 0.0038948
5: 0.0060399, 0.0106749, 0.0063696, 0.0104644, -0.0033604, 0.0029855
6: 0.0070816, 0.0106536, 0.0075264, 0.0105292, -0.0034476, 0.0031272
7: -0.0215735, -0.0115115, -0.0211164, -0.0122271, -0.0055851, 0.0057041
8: 0.9619802, 0.9908092, 0.9632899, 0.9887587, -0.0197312, 0.0222744
9: 0.0003543, 0.0088272, 0.0009570, 0.0084423, -0.0052708, 0.0048979

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1_B2_B2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118867, upper bound: 0.0110280
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B2_B2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118867, upper bound: 0.0110280
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0004980, 0.0011645, -0.0004669, 0.0009589, -0.0010866, 0.0012141
1: -0.0010110, 0.0031062, -0.0008653, 0.0027912, -0.0027995, 0.0032168
2: 0.0116881, 0.0178541, 0.0121599, 0.0176359, -0.0044992, 0.0039074
3: -0.0018380, 0.0027986, -0.0014832, 0.0026346, -0.0032375, 0.0028161
4: -0.0060750, -0.0017982, -0.0057477, -0.0019495, -0.0040743, 0.0036696
5: 0.0061036, 0.0107317, 0.0064577, 0.0105679, -0.0032177, 0.0027997
6: 0.0071675, 0.0106296, 0.0076453, 0.0104959, -0.0033284, 0.0029843
7: -0.0216967, -0.0116497, -0.0213413, -0.0124185, -0.0052295, 0.0053927
8: 0.9616272, 0.9904132, 0.9626456, 0.9882106, -0.0185265, 0.0213351
9: 0.0004707, 0.0089309, 0.0011181, 0.0086316, -0.0050163, 0.0045810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A2_A1_B1_B1_B1

### Relational analysis result of IS_A2_B1_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120847, upper bound: 0.0112412
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_A1_B1_B1_B2

### Relational analysis result of IS_A2_B1_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120847, upper bound: 0.0112412
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0004982, 0.0011695, -0.0004619, 0.0009633, -0.0010898, 0.0012235
1: -0.0010118, 0.0031139, -0.0008421, 0.0027979, -0.0028084, 0.0032370
2: 0.0116766, 0.0178553, 0.0121498, 0.0176011, -0.0045322, 0.0039193
3: -0.0018466, 0.0027995, -0.0014908, 0.0026084, -0.0032645, 0.0028231
4: -0.0060829, -0.0017973, -0.0057547, -0.0019737, -0.0040948, 0.0036796
5: 0.0060949, 0.0107326, 0.0064501, 0.0105418, -0.0032447, 0.0028066
6: 0.0071558, 0.0106328, 0.0076350, 0.0104988, -0.0033430, 0.0029978
7: -0.0216987, -0.0116310, -0.0212844, -0.0124020, -0.0052394, 0.0054656
8: 0.9616215, 0.9904668, 0.9628084, 0.9882579, -0.0185841, 0.0214837
9: 0.0004549, 0.0089326, 0.0011042, 0.0085838, -0.0050735, 0.0045846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A2_A1_B1_B2_B1

### Relational analysis result of IS_A2_B1_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120829, upper bound: 0.0112505
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A2_A1_B1_B2_B2

### Relational analysis result of IS_A2_B1_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120829, upper bound: 0.0112505
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0004980, 0.0011645, -0.0004415, 0.0010525, -0.0011884, 0.0012172
1: -0.0010110, 0.0031062, -0.0007464, 0.0029346, -0.0030357, 0.0032610
2: 0.0116881, 0.0178541, 0.0119451, 0.0174578, -0.0045450, 0.0042611
3: -0.0018380, 0.0027986, -0.0016447, 0.0025006, -0.0032628, 0.0030822
4: -0.0060750, -0.0017982, -0.0058967, -0.0020730, -0.0040019, 0.0039150
5: 0.0061036, 0.0107317, 0.0062965, 0.0104342, -0.0032419, 0.0030652
6: 0.0071675, 0.0106296, 0.0074278, 0.0105568, -0.0033893, 0.0032018
7: -0.0216967, -0.0116497, -0.0210510, -0.0120685, -0.0058059, 0.0054549
8: 0.9616272, 0.9904132, 0.9634772, 0.9892132, -0.0201780, 0.0215695
9: 0.0004707, 0.0089309, 0.0008234, 0.0083872, -0.0050465, 0.0050664

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A2_A1_B2_B1_B1

### Relational analysis result of IS_A2_B1_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118740, upper bound: 0.0110190
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A2_A1_B2_B1_B2

### Relational analysis result of IS_A2_B1_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118740, upper bound: 0.0110190
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0004982, 0.0011695, -0.0004365, 0.0010577, -0.0011934, 0.0012249
1: -0.0010118, 0.0031139, -0.0007231, 0.0029425, -0.0030470, 0.0032818
2: 0.0116766, 0.0178553, 0.0119333, 0.0174229, -0.0045764, 0.0042767
3: -0.0018466, 0.0027995, -0.0016536, 0.0024744, -0.0032877, 0.0030918
4: -0.0060829, -0.0017973, -0.0059049, -0.0020973, -0.0039857, 0.0039275
5: 0.0060949, 0.0107326, 0.0062876, 0.0104081, -0.0032672, 0.0030749
6: 0.0071558, 0.0106328, 0.0074158, 0.0105601, -0.0034043, 0.0032171
7: -0.0216987, -0.0116310, -0.0209942, -0.0120492, -0.0058217, 0.0055073
8: 0.9616215, 0.9904668, 0.9636401, 0.9892687, -0.0202524, 0.0217148
9: 0.0004549, 0.0089326, 0.0008071, 0.0083394, -0.0050990, 0.0050750

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A2_A1_B2_B2_B1

### Relational analysis result of IS_A2_B1_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118770, upper bound: 0.0110321
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_A1_B2_B2_B2

### Relational analysis result of IS_A2_B1_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118770, upper bound: 0.0110321
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004783, 0.0012478, -0.0004621, 0.0009805, -0.0010966, 0.0013076
1: -0.0009188, 0.0032339, -0.0008430, 0.0028242, -0.0028541, 0.0034407
2: 0.0114969, 0.0177161, 0.0121104, 0.0176024, -0.0048311, 0.0039618
3: -0.0019817, 0.0026948, -0.0015204, 0.0026094, -0.0034862, 0.0028516
4: -0.0062076, -0.0018939, -0.0057820, -0.0019728, -0.0042348, 0.0037901
5: 0.0059600, 0.0106281, 0.0064206, 0.0105428, -0.0034657, 0.0028346
6: 0.0069738, 0.0106837, 0.0075951, 0.0105100, -0.0035361, 0.0030886
7: -0.0214718, -0.0113381, -0.0212866, -0.0123378, -0.0052048, 0.0059033
8: 0.9622716, 0.9913060, 0.9628021, 0.9884416, -0.0188002, 0.0228863
9: 0.0002083, 0.0087416, 0.0010502, 0.0085856, -0.0054572, 0.0046012

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A2_A2_A1_B1_B1

### Relational analysis result of IS_A2_B1_A2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118872, upper bound: 0.0110459
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_A2_A1_B1_B2

### Relational analysis result of IS_A2_B1_A2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118872, upper bound: 0.0110459
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004783, 0.0012478, -0.0004367, 0.0010733, -0.0011809, 0.0012755
1: -0.0009188, 0.0032339, -0.0007240, 0.0029665, -0.0030010, 0.0033916
2: 0.0114969, 0.0177161, 0.0118974, 0.0174243, -0.0047239, 0.0041679
3: -0.0019817, 0.0026948, -0.0016806, 0.0024754, -0.0033886, 0.0029961
4: -0.0062076, -0.0018939, -0.0059298, -0.0020963, -0.0041113, 0.0039508
5: 0.0059600, 0.0106281, 0.0062607, 0.0104091, -0.0033669, 0.0029777
6: 0.0069738, 0.0106837, 0.0073794, 0.0105703, -0.0035965, 0.0033043
7: -0.0214718, -0.0113381, -0.0209964, -0.0119908, -0.0054665, 0.0054883
8: 0.9622716, 0.9913060, 0.9636337, 0.9894360, -0.0197809, 0.0224240
9: 0.0002083, 0.0087416, 0.0007579, 0.0083412, -0.0051876, 0.0047989

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A2_A2_A1_B2_B1

### Relational analysis result of IS_A2_B1_A2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118872, upper bound: 0.0110459
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_A2_A1_B2_B2

### Relational analysis result of IS_A2_B1_A2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118872, upper bound: 0.0110459
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004766, 0.0012603, -0.0004622, 0.0009842, -0.0011012, 0.0013123
1: -0.0009108, 0.0032530, -0.0008435, 0.0028300, -0.0028839, 0.0034378
2: 0.0114682, 0.0177040, 0.0121018, 0.0176032, -0.0048207, 0.0039975
3: -0.0020033, 0.0026858, -0.0015269, 0.0026100, -0.0034754, 0.0028782
4: -0.0062275, -0.0019023, -0.0057880, -0.0019722, -0.0042553, 0.0038312
5: 0.0059385, 0.0106190, 0.0064141, 0.0105434, -0.0034548, 0.0028613
6: 0.0069448, 0.0106918, 0.0075864, 0.0105124, -0.0035676, 0.0031054
7: -0.0214522, -0.0112915, -0.0212880, -0.0123238, -0.0052346, 0.0058545
8: 0.9623277, 0.9914396, 0.9627982, 0.9884818, -0.0189718, 0.0228445
9: 0.0001690, 0.0087251, 0.0010383, 0.0085868, -0.0054257, 0.0046358

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A2_A2_A2_B1_B1

### Relational analysis result of IS_A2_B1_A2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118935, upper bound: 0.0110318
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_A2_A2_B1_B2

### Relational analysis result of IS_A2_B1_A2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118935, upper bound: 0.0110318
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004766, 0.0012603, -0.0004368, 0.0010770, -0.0011866, 0.0012806
1: -0.0009108, 0.0032530, -0.0007246, 0.0029721, -0.0030325, 0.0033902
2: 0.0114682, 0.0177040, 0.0118889, 0.0174252, -0.0047186, 0.0042102
3: -0.0020033, 0.0026858, -0.0016869, 0.0024761, -0.0033823, 0.0030254
4: -0.0062275, -0.0019023, -0.0059357, -0.0020957, -0.0041318, 0.0039929
5: 0.0059385, 0.0106190, 0.0062543, 0.0104097, -0.0033602, 0.0030069
6: 0.0069448, 0.0106918, 0.0073709, 0.0105727, -0.0036279, 0.0033210
7: -0.0214522, -0.0112915, -0.0209978, -0.0119769, -0.0055059, 0.0054377
8: 0.9623277, 0.9914396, 0.9636295, 0.9894756, -0.0199858, 0.0224026
9: 0.0001690, 0.0087251, 0.0007463, 0.0083424, -0.0051595, 0.0048457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A2_A2_A2_B2_B1

### Relational analysis result of IS_A2_B1_A2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118935, upper bound: 0.0110318
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_A2_A2_B2_B2

### Relational analysis result of IS_A2_B1_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118935, upper bound: 0.0110318
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0004870, 0.0012076, -0.0004870, 0.0012076, -0.0012525, 0.0012525
1: -0.0009594, 0.0031724, -0.0009594, 0.0031724, -0.0031531, 0.0031531
2: 0.0115890, 0.0177767, 0.0115890, 0.0177767, -0.0043458, 0.0043458
3: -0.0019124, 0.0027405, -0.0019124, 0.0027405, -0.0030965, 0.0030965
4: -0.0061437, -0.0018518, -0.0061437, -0.0018518, -0.0041812, 0.0041812
5: 0.0060292, 0.0106736, 0.0060292, 0.0106736, -0.0030745, 0.0030745
6: 0.0070672, 0.0106576, 0.0070672, 0.0106576, -0.0035905, 0.0035905
7: -0.0215707, -0.0114883, -0.0215707, -0.0114883, -0.0052073, 0.0052073
8: 0.9619883, 0.9908757, 0.9619883, 0.9908757, -0.0206785, 0.0206785
9: 0.0003348, 0.0088248, 0.0003348, 0.0088248, -0.0046970, 0.0046970

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A2_B2_A1_A1_B1_B1_A1

### Relational analysis result of IS_A2_B2_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120852, upper bound: 0.0112549
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_A1_B1_B1_A2

### Relational analysis result of IS_A2_B2_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120872, upper bound: 0.0112280
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0004870, 0.0012076, -0.0004983, 0.0011751, -0.0012225, 0.0012672
1: -0.0009594, 0.0031724, -0.0010123, 0.0031225, -0.0031082, 0.0031479
2: 0.0115890, 0.0177767, 0.0116637, 0.0178561, -0.0043585, 0.0042786
3: -0.0019124, 0.0027405, -0.0018563, 0.0028001, -0.0031144, 0.0030459
4: -0.0061437, -0.0018518, -0.0060918, -0.0017968, -0.0041365, 0.0041346
5: 0.0060292, 0.0106736, 0.0060853, 0.0107332, -0.0030934, 0.0030240
6: 0.0070672, 0.0106576, 0.0071428, 0.0106365, -0.0035693, 0.0035148
7: -0.0215707, -0.0114883, -0.0217000, -0.0116100, -0.0050977, 0.0053853
8: 0.9619883, 0.9908757, 0.9616177, 0.9905269, -0.0203644, 0.0207178
9: 0.0003348, 0.0088248, 0.0004373, 0.0089337, -0.0048154, 0.0046047

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A2_B2_A1_A1_B1_B2_A1

### Relational analysis result of IS_A2_B2_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120852, upper bound: 0.0112549
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_A1_B1_B2_A2

### Relational analysis result of IS_A2_B2_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120872, upper bound: 0.0112280
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0004870, 0.0012076, -0.0004653, 0.0013117, -0.0013649, 0.0012484
1: -0.0009594, 0.0031724, -0.0008580, 0.0033318, -0.0033912, 0.0032269
2: 0.0115890, 0.0177767, 0.0113502, 0.0176249, -0.0044207, 0.0047024
3: -0.0019124, 0.0027405, -0.0020920, 0.0026263, -0.0031399, 0.0033646
4: -0.0061437, -0.0018518, -0.0063093, -0.0019571, -0.0041865, 0.0044286
5: 0.0060292, 0.0106736, 0.0058500, 0.0105597, -0.0031168, 0.0033421
6: 0.0070672, 0.0106576, 0.0068253, 0.0107253, -0.0036581, 0.0038323
7: -0.0215707, -0.0114883, -0.0213233, -0.0110992, -0.0057884, 0.0052114
8: 0.9619883, 0.9908757, 0.9626971, 0.9919905, -0.0223432, 0.0210574
9: 0.0003348, 0.0088248, 0.0000071, 0.0086165, -0.0047521, 0.0051863

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A2_B2_A1_A1_B2_B1_B1

### Relational analysis result of IS_A2_B2_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118811, upper bound: 0.0110213
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2_B1_B2

### Relational analysis result of IS_A2_B2_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118836, upper bound: 0.0110359
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0004870, 0.0012076, -0.0004771, 0.0012833, -0.0013349, 0.0012587
1: -0.0009594, 0.0031724, -0.0009130, 0.0032884, -0.0033349, 0.0032190
2: 0.0115890, 0.0177767, 0.0114153, 0.0177073, -0.0044269, 0.0046182
3: -0.0019124, 0.0027405, -0.0020431, 0.0026883, -0.0031532, 0.0033013
4: -0.0061437, -0.0018518, -0.0062642, -0.0019000, -0.0042437, 0.0043701
5: 0.0060292, 0.0106736, 0.0058988, 0.0106215, -0.0031313, 0.0032789
6: 0.0070672, 0.0106576, 0.0068912, 0.0107069, -0.0036397, 0.0037665
7: -0.0215707, -0.0114883, -0.0214576, -0.0112051, -0.0056511, 0.0053301
8: 0.9619883, 0.9908757, 0.9623121, 0.9916869, -0.0219500, 0.0210700
9: 0.0003348, 0.0088248, 0.0000963, 0.0087296, -0.0048369, 0.0050707

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A2_B2_A1_A1_B2_B2_B1

### Relational analysis result of IS_A2_B2_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118811, upper bound: 0.0110213
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2_B2_B2

### Relational analysis result of IS_A2_B2_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118836, upper bound: 0.0110359
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004671, 0.0012760, -0.0004870, 0.0011969, -0.0012386, 0.0013336
1: -0.0008664, 0.0032772, -0.0009597, 0.0031559, -0.0032326, 0.0033195
2: 0.0114321, 0.0176375, 0.0116137, 0.0177772, -0.0045988, 0.0044297
3: -0.0020305, 0.0026358, -0.0018939, 0.0027408, -0.0032868, 0.0031465
4: -0.0062525, -0.0019484, -0.0061265, -0.0018515, -0.0043420, 0.0041781
5: 0.0059114, 0.0105691, 0.0060478, 0.0106740, -0.0032644, 0.0031232
6: 0.0069082, 0.0107021, 0.0070922, 0.0106506, -0.0037424, 0.0036099
7: -0.0213438, -0.0112325, -0.0215715, -0.0115285, -0.0051414, 0.0057054
8: 0.9626384, 0.9916085, 0.9619861, 0.9907603, -0.0211006, 0.0218569
9: 0.0001194, 0.0086338, 0.0003687, 0.0088255, -0.0050995, 0.0046906

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119859, upper bound: 0.0111347
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119859, upper bound: 0.0111347
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004648, 0.0012888, -0.0004872, 0.0012014, -0.0012440, 0.0013443
1: -0.0008558, 0.0032969, -0.0009605, 0.0031629, -0.0032683, 0.0033339
2: 0.0114026, 0.0176216, 0.0116033, 0.0177785, -0.0046161, 0.0044827
3: -0.0020526, 0.0026238, -0.0019017, 0.0027417, -0.0033009, 0.0031850
4: -0.0062730, -0.0019595, -0.0061338, -0.0018506, -0.0043647, 0.0041743
5: 0.0058893, 0.0105572, 0.0060399, 0.0106749, -0.0032787, 0.0031615
6: 0.0068784, 0.0107104, 0.0070816, 0.0106536, -0.0037752, 0.0036288
7: -0.0213179, -0.0111845, -0.0215735, -0.0115115, -0.0051859, 0.0057228
8: 0.9627127, 0.9917461, 0.9619802, 0.9908092, -0.0213544, 0.0219417
9: 0.0000790, 0.0086119, 0.0003543, 0.0088272, -0.0051109, 0.0047452

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119869, upper bound: 0.0110971
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119869, upper bound: 0.0110971
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004671, 0.0012760, -0.0004985, 0.0011640, -0.0012084, 0.0013463
1: -0.0008664, 0.0032772, -0.0010134, 0.0031055, -0.0031878, 0.0033143
2: 0.0114321, 0.0176375, 0.0116891, 0.0178577, -0.0046067, 0.0043625
3: -0.0020305, 0.0026358, -0.0018372, 0.0028013, -0.0033018, 0.0030960
4: -0.0062525, -0.0019484, -0.0060743, -0.0017957, -0.0042976, 0.0041258
5: 0.0059114, 0.0105691, 0.0061043, 0.0107344, -0.0032806, 0.0030728
6: 0.0069082, 0.0107021, 0.0071685, 0.0106293, -0.0037211, 0.0035336
7: -0.0213438, -0.0112325, -0.0217026, -0.0116513, -0.0050320, 0.0058569
8: 0.9626384, 0.9916085, 0.9616104, 0.9904085, -0.0207871, 0.0218761
9: 0.0001194, 0.0086338, 0.0004721, 0.0089359, -0.0052097, 0.0045984

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118996, upper bound: 0.0110570
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118996, upper bound: 0.0110570
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004648, 0.0012888, -0.0004987, 0.0011690, -0.0012142, 0.0013559
1: -0.0008558, 0.0032969, -0.0010142, 0.0031133, -0.0032233, 0.0033301
2: 0.0114026, 0.0176216, 0.0116776, 0.0178588, -0.0046266, 0.0044153
3: -0.0020526, 0.0026238, -0.0018459, 0.0028022, -0.0033166, 0.0031343
4: -0.0062730, -0.0019595, -0.0060823, -0.0017949, -0.0043213, 0.0041228
5: 0.0058893, 0.0105572, 0.0060957, 0.0107352, -0.0032952, 0.0031109
6: 0.0068784, 0.0107104, 0.0071568, 0.0106326, -0.0037542, 0.0035536
7: -0.0213179, -0.0111845, -0.0217044, -0.0116325, -0.0050760, 0.0058624
8: 0.9627127, 0.9917461, 0.9616051, 0.9904624, -0.0210395, 0.0219730
9: 0.0000790, 0.0086119, 0.0004562, 0.0089374, -0.0052156, 0.0046527

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119031, upper bound: 0.0110359
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119031, upper bound: 0.0110359
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0004980, 0.0011645, -0.0004904, 0.0011897, -0.0012483, 0.0012181
1: -0.0010110, 0.0031062, -0.0009757, 0.0031448, -0.0030952, 0.0030917
2: 0.0116881, 0.0178541, 0.0116303, 0.0178013, -0.0042553, 0.0042819
3: -0.0018380, 0.0027986, -0.0018814, 0.0027589, -0.0030301, 0.0030584
4: -0.0060750, -0.0017982, -0.0061151, -0.0018348, -0.0041081, 0.0040730
5: 0.0061036, 0.0107317, 0.0060602, 0.0106920, -0.0030082, 0.0030376
6: 0.0071675, 0.0106296, 0.0071089, 0.0106459, -0.0034785, 0.0035206
7: -0.0216967, -0.0116497, -0.0216107, -0.0115555, -0.0052888, 0.0051203
8: 0.9616272, 0.9904132, 0.9618737, 0.9906831, -0.0203584, 0.0202559
9: 0.0004707, 0.0089309, 0.0003914, 0.0088585, -0.0046096, 0.0047262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A2_A1_B1_B1_B1

### Relational analysis result of IS_A2_B2_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120847, upper bound: 0.0112441
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_A1_B1_B1_B2

### Relational analysis result of IS_A2_B2_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120847, upper bound: 0.0112441
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0004982, 0.0011695, -0.0004870, 0.0011968, -0.0012542, 0.0012240
1: -0.0010118, 0.0031139, -0.0009595, 0.0031558, -0.0031120, 0.0031255
2: 0.0116766, 0.0178553, 0.0116138, 0.0177769, -0.0043050, 0.0043011
3: -0.0018466, 0.0027995, -0.0018938, 0.0027406, -0.0030664, 0.0030708
4: -0.0060829, -0.0017973, -0.0061265, -0.0018517, -0.0041444, 0.0040965
5: 0.0060949, 0.0107326, 0.0060478, 0.0106738, -0.0030445, 0.0030496
6: 0.0071558, 0.0106328, 0.0070922, 0.0106506, -0.0034948, 0.0035406
7: -0.0216987, -0.0116310, -0.0215710, -0.0115286, -0.0052877, 0.0051597
8: 0.9616215, 0.9904668, 0.9619873, 0.9907601, -0.0204532, 0.0204871
9: 0.0004549, 0.0089326, 0.0003687, 0.0088251, -0.0046533, 0.0047297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A2_A1_B1_B2_B1

### Relational analysis result of IS_A2_B2_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120829, upper bound: 0.0112542
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_A1_B1_B2_B2

### Relational analysis result of IS_A2_B2_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120829, upper bound: 0.0112542
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0004980, 0.0011645, -0.0004676, 0.0012934, -0.0013571, 0.0012143
1: -0.0010110, 0.0031062, -0.0008686, 0.0033038, -0.0033274, 0.0031700
2: 0.0116881, 0.0178541, 0.0113921, 0.0176407, -0.0043408, 0.0046296
3: -0.0018380, 0.0027986, -0.0020605, 0.0026382, -0.0030807, 0.0033198
4: -0.0060750, -0.0017982, -0.0062802, -0.0019462, -0.0041288, 0.0043141
5: 0.0061036, 0.0107317, 0.0058814, 0.0105715, -0.0030579, 0.0032985
6: 0.0071675, 0.0106296, 0.0068678, 0.0107134, -0.0035459, 0.0037618
7: -0.0216967, -0.0116497, -0.0213491, -0.0111675, -0.0058553, 0.0051285
8: 0.9616272, 0.9904132, 0.9626232, 0.9917949, -0.0219813, 0.0206805
9: 0.0004707, 0.0089309, 0.0000646, 0.0086382, -0.0046612, 0.0052032

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A2_A1_B2_B1_B1

### Relational analysis result of IS_A2_B2_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118740, upper bound: 0.0110315
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2_B1_B2

### Relational analysis result of IS_A2_B2_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118740, upper bound: 0.0110315
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0004982, 0.0011695, -0.0004653, 0.0013054, -0.0013671, 0.0012198
1: -0.0010118, 0.0031139, -0.0008582, 0.0033222, -0.0033440, 0.0032025
2: 0.0116766, 0.0178553, 0.0113646, 0.0176252, -0.0043883, 0.0046487
3: -0.0018466, 0.0027995, -0.0020812, 0.0026265, -0.0031157, 0.0033322
4: -0.0060829, -0.0017973, -0.0062993, -0.0019569, -0.0041260, 0.0043376
5: 0.0060949, 0.0107326, 0.0058608, 0.0105599, -0.0030925, 0.0033105
6: 0.0071558, 0.0106328, 0.0068399, 0.0107212, -0.0035654, 0.0037929
7: -0.0216987, -0.0116310, -0.0213238, -0.0111226, -0.0058540, 0.0051629
8: 0.9616215, 0.9904668, 0.9626956, 0.9919235, -0.0220759, 0.0209046
9: 0.0004549, 0.0089326, 0.0000268, 0.0086169, -0.0047104, 0.0052066

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A2_A1_B2_B2_B1

### Relational analysis result of IS_A2_B2_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118770, upper bound: 0.0110433
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2_B2_B2

### Relational analysis result of IS_A2_B2_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118770, upper bound: 0.0110433
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004783, 0.0012478, -0.0004872, 0.0012120, -0.0012627, 0.0013043
1: -0.0009188, 0.0032339, -0.0009606, 0.0031790, -0.0031728, 0.0033234
2: 0.0114969, 0.0177161, 0.0115791, 0.0177786, -0.0045975, 0.0043662
3: -0.0019817, 0.0026948, -0.0019199, 0.0027418, -0.0032848, 0.0031123
4: -0.0062076, -0.0018939, -0.0061506, -0.0018506, -0.0043570, 0.0042154
5: 0.0059600, 0.0106281, 0.0060217, 0.0106750, -0.0032624, 0.0030907
6: 0.0069738, 0.0106837, 0.0070571, 0.0106605, -0.0036866, 0.0036267
7: -0.0214718, -0.0113381, -0.0215737, -0.0114721, -0.0053172, 0.0056075
8: 0.9622716, 0.9913060, 0.9619798, 0.9909220, -0.0207802, 0.0218562
9: 0.0002083, 0.0087416, 0.0003211, 0.0088273, -0.0050374, 0.0047999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A2_A2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118872, upper bound: 0.0110588
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A2_A2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118872, upper bound: 0.0110588
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004783, 0.0012478, -0.0004656, 0.0013180, -0.0013527, 0.0012779
1: -0.0009188, 0.0032339, -0.0008592, 0.0033415, -0.0033258, 0.0033191
2: 0.0114969, 0.0177161, 0.0113358, 0.0176267, -0.0045487, 0.0045881
3: -0.0019817, 0.0026948, -0.0021029, 0.0026277, -0.0032302, 0.0032730
4: -0.0062076, -0.0018939, -0.0063193, -0.0019559, -0.0042517, 0.0043837
5: 0.0059600, 0.0106281, 0.0058391, 0.0105610, -0.0032061, 0.0032503
6: 0.0069738, 0.0106837, 0.0068107, 0.0107294, -0.0037556, 0.0038730
7: -0.0214718, -0.0113381, -0.0213263, -0.0110756, -0.0055126, 0.0052266
8: 0.9622716, 0.9913060, 0.9626885, 0.9920580, -0.0218195, 0.0216731
9: 0.0002083, 0.0087416, -0.0000127, 0.0086190, -0.0048008, 0.0049865

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A2_A2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118872, upper bound: 0.0110588
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_A2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118872, upper bound: 0.0110588
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004766, 0.0012603, -0.0004874, 0.0012164, -0.0012688, 0.0013161
1: -0.0009108, 0.0032530, -0.0009614, 0.0031858, -0.0032064, 0.0033385
2: 0.0114682, 0.0177040, 0.0115689, 0.0177798, -0.0046173, 0.0044149
3: -0.0020033, 0.0026858, -0.0019276, 0.0027428, -0.0032984, 0.0031477
4: -0.0062275, -0.0019023, -0.0061576, -0.0018497, -0.0043778, 0.0042553
5: 0.0059385, 0.0106190, 0.0060141, 0.0106760, -0.0032758, 0.0031259
6: 0.0069448, 0.0106918, 0.0070468, 0.0106633, -0.0037185, 0.0036451
7: -0.0214522, -0.0112915, -0.0215758, -0.0114555, -0.0053600, 0.0056202
8: 0.9623277, 0.9914396, 0.9619738, 0.9909696, -0.0210097, 0.0219510
9: 0.0001690, 0.0087251, 0.0003072, 0.0088291, -0.0050550, 0.0048503

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A2_A2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118935, upper bound: 0.0110433
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_A2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118935, upper bound: 0.0110433
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004766, 0.0012603, -0.0004657, 0.0013237, -0.0013616, 0.0012896
1: -0.0009108, 0.0032530, -0.0008600, 0.0033502, -0.0033582, 0.0033357
2: 0.0114682, 0.0177040, 0.0113227, 0.0176279, -0.0045689, 0.0046375
3: -0.0020033, 0.0026858, -0.0021127, 0.0026285, -0.0032439, 0.0033108
4: -0.0062275, -0.0019023, -0.0063284, -0.0019551, -0.0042724, 0.0044233
5: 0.0059385, 0.0106190, 0.0058293, 0.0105619, -0.0032197, 0.0032878
6: 0.0069448, 0.0106918, 0.0067975, 0.0107331, -0.0037882, 0.0038944
7: -0.0214522, -0.0112915, -0.0213281, -0.0110544, -0.0055699, 0.0052406
8: 0.9623277, 0.9914396, 0.9626833, 0.9921188, -0.0220488, 0.0217698
9: 0.0001690, 0.0087251, -0.0000306, 0.0086206, -0.0048126, 0.0050509

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A2_A2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118935, upper bound: 0.0110433
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_A2_A2_B2_B2

### Relational analysis result of IS_A2_B2_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118935, upper bound: 0.0110433
time: 0.78 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.32 seconds
IS_A1_B1_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0114753, upper bound: 0.0115048
IS_A1_B1_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0114857, upper bound: 0.0114855
IS_A1_B1_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0114753, upper bound: 0.0115048
IS_A1_B1_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0114857, upper bound: 0.0114855
IS_A1_B1_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0112938, upper bound: 0.0112716
IS_A1_B1_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0112952, upper bound: 0.0112906
IS_A1_B1_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0112938, upper bound: 0.0112716
IS_A1_B1_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0112952, upper bound: 0.0112906
IS_A1_B1_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0113562, upper bound: 0.0113686
IS_A1_B1_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0113562, upper bound: 0.0113686
IS_A1_B1_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0113628, upper bound: 0.0113577
IS_A1_B1_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0113628, upper bound: 0.0113577
IS_A1_B1_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0112659, upper bound: 0.0112829
IS_A1_B1_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0112659, upper bound: 0.0112829
IS_A1_B1_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0112781, upper bound: 0.0112837
IS_A1_B1_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0112781, upper bound: 0.0112837
IS_A1_B1_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0115033, upper bound: 0.0114888
IS_A1_B1_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0115033, upper bound: 0.0114888
IS_A1_B1_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0114852, upper bound: 0.0114943
IS_A1_B1_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0114852, upper bound: 0.0114943
IS_A1_B1_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0112923, upper bound: 0.0112711
IS_A1_B1_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0112923, upper bound: 0.0112711
IS_A1_B1_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0112923, upper bound: 0.0112860
IS_A1_B1_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0112923, upper bound: 0.0112860
IS_A1_B1_A2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0112602, upper bound: 0.0112740
IS_A1_B1_A2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0112602, upper bound: 0.0112740
IS_A1_B1_A2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0112602, upper bound: 0.0112740
IS_A1_B1_A2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0112602, upper bound: 0.0112740
IS_A1_B1_A2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0112716, upper bound: 0.0112736
IS_A1_B1_A2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0112716, upper bound: 0.0112736
IS_A1_B1_A2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0112716, upper bound: 0.0112736
IS_A1_B1_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0112716, upper bound: 0.0112736
IS_A1_B2_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0112113, upper bound: 0.0120867
IS_A1_B2_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0112234, upper bound: 0.0120872
IS_A1_B2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0112531, upper bound: 0.0120935
IS_A1_B2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0112234, upper bound: 0.0120963
IS_A1_B2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0110740, upper bound: 0.0119194
IS_A1_B2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0110699, upper bound: 0.0119232
IS_A1_B2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0110740, upper bound: 0.0119227
IS_A1_B2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0110699, upper bound: 0.0119281
IS_A1_B2_B1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0110800, upper bound: 0.0119556
IS_A1_B2_B1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0110800, upper bound: 0.0119556
IS_A1_B2_B1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0110920, upper bound: 0.0119510
IS_A1_B2_B1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0110920, upper bound: 0.0119510
IS_A1_B2_B1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0110084, upper bound: 0.0118820
IS_A1_B2_B1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0110084, upper bound: 0.0118820
IS_A1_B2_B1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0110280, upper bound: 0.0118867
IS_A1_B2_B1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0110280, upper bound: 0.0118867
IS_A1_B2_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0112113, upper bound: 0.0120847
IS_A1_B2_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0112113, upper bound: 0.0120921
IS_A1_B2_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0112234, upper bound: 0.0120829
IS_A1_B2_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0112234, upper bound: 0.0120963
IS_A1_B2_B2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0110052, upper bound: 0.0118740
IS_A1_B2_B2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0110052, upper bound: 0.0118806
IS_A1_B2_B2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0110232, upper bound: 0.0118770
IS_A1_B2_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0110232, upper bound: 0.0118857
IS_A1_B2_B2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0110385, upper bound: 0.0118872
IS_A1_B2_B2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0110385, upper bound: 0.0118923
IS_A1_B2_B2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0110385, upper bound: 0.0118679
IS_A1_B2_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0110385, upper bound: 0.0118761
IS_A1_B2_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0110232, upper bound: 0.0118935
IS_A1_B2_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0110232, upper bound: 0.0118975
IS_A1_B2_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0110232, upper bound: 0.0118770
IS_A1_B2_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0110232, upper bound: 0.0118845
IS_A2_B1_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0120867, upper bound: 0.0112113
IS_A2_B1_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0120872, upper bound: 0.0112234
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0120852, upper bound: 0.0112531
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0120872, upper bound: 0.0112234
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0119194, upper bound: 0.0110740
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0119232, upper bound: 0.0110699
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0119194, upper bound: 0.0110740
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0119232, upper bound: 0.0110699
IS_A2_B1_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0119556, upper bound: 0.0110800
IS_A2_B1_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0119556, upper bound: 0.0110796
IS_A2_B1_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0119510, upper bound: 0.0110920
IS_A2_B1_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0119510, upper bound: 0.0110920
IS_A2_B1_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0118820, upper bound: 0.0110084
IS_A2_B1_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0118820, upper bound: 0.0110084
IS_A2_B1_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0118867, upper bound: 0.0110280
IS_A2_B1_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0118867, upper bound: 0.0110280
IS_A2_B1_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0120847, upper bound: 0.0112412
IS_A2_B1_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0120847, upper bound: 0.0112412
IS_A2_B1_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0120829, upper bound: 0.0112505
IS_A2_B1_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0120829, upper bound: 0.0112505
IS_A2_B1_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0118740, upper bound: 0.0110190
IS_A2_B1_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0118740, upper bound: 0.0110190
IS_A2_B1_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0118770, upper bound: 0.0110321
IS_A2_B1_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0118770, upper bound: 0.0110321
IS_A2_B1_A2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0118872, upper bound: 0.0110459
IS_A2_B1_A2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0118872, upper bound: 0.0110459
IS_A2_B1_A2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0118872, upper bound: 0.0110459
IS_A2_B1_A2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0118872, upper bound: 0.0110459
IS_A2_B1_A2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0118935, upper bound: 0.0110318
IS_A2_B1_A2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0118935, upper bound: 0.0110318
IS_A2_B1_A2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0118935, upper bound: 0.0110318
IS_A2_B1_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0118935, upper bound: 0.0110318
IS_A2_B2_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0120852, upper bound: 0.0112549
IS_A2_B2_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0120872, upper bound: 0.0112280
IS_A2_B2_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0120852, upper bound: 0.0112549
IS_A2_B2_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0120872, upper bound: 0.0112280
IS_A2_B2_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0118811, upper bound: 0.0110213
IS_A2_B2_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0118836, upper bound: 0.0110359
IS_A2_B2_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0118811, upper bound: 0.0110213
IS_A2_B2_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0118836, upper bound: 0.0110359
IS_A2_B2_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0119859, upper bound: 0.0111347
IS_A2_B2_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0119859, upper bound: 0.0111347
IS_A2_B2_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0119869, upper bound: 0.0110971
IS_A2_B2_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0119869, upper bound: 0.0110971
IS_A2_B2_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0118996, upper bound: 0.0110570
IS_A2_B2_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0118996, upper bound: 0.0110570
IS_A2_B2_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0119031, upper bound: 0.0110359
IS_A2_B2_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0119031, upper bound: 0.0110359
IS_A2_B2_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0120847, upper bound: 0.0112441
IS_A2_B2_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0120847, upper bound: 0.0112441
IS_A2_B2_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0120829, upper bound: 0.0112542
IS_A2_B2_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0120829, upper bound: 0.0112542
IS_A2_B2_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0118740, upper bound: 0.0110315
IS_A2_B2_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0118740, upper bound: 0.0110315
IS_A2_B2_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0118770, upper bound: 0.0110433
IS_A2_B2_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0118770, upper bound: 0.0110433
IS_A2_B2_A2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0118872, upper bound: 0.0110588
IS_A2_B2_A2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0118872, upper bound: 0.0110588
IS_A2_B2_A2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0118872, upper bound: 0.0110588
IS_A2_B2_A2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0118872, upper bound: 0.0110588
IS_A2_B2_A2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0118935, upper bound: 0.0110433
IS_A2_B2_A2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0118935, upper bound: 0.0110433
IS_A2_B2_A2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0118935, upper bound: 0.0110433
IS_A2_B2_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 8, lower bound: -0.0118935, upper bound: 0.0110433

## BFS IS instance: IS_A1_B1_A1_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004665, 0.0009445, -0.0004617, 0.0009662, -0.0009864, 0.0009684
1: -0.0008635, 0.0027691, -0.0008412, 0.0028024, -0.0027015, 0.0027003
2: 0.0121930, 0.0176331, 0.0121431, 0.0175998, -0.0037269, 0.0037322
3: -0.0014583, 0.0026325, -0.0014958, 0.0026074, -0.0026581, 0.0026629
4: -0.0057248, -0.0019515, -0.0057593, -0.0019746, -0.0035391, 0.0035294
5: 0.0064825, 0.0105658, 0.0064451, 0.0105408, -0.0026394, 0.0026444
6: 0.0076788, 0.0104866, 0.0076283, 0.0105007, -0.0028219, 0.0028583
7: -0.0213367, -0.0124724, -0.0212824, -0.0123911, -0.0041822, 0.0041434
8: 0.9626588, 0.9880561, 0.9628144, 0.9882889, -0.0177485, 0.0177266
9: 0.0011635, 0.0086278, 0.0010950, 0.0085820, -0.0039623, 0.0039855

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A1_A1_B1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114318, upper bound: 0.0114619
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114302, upper bound: 0.0114619
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004615, 0.0009492, -0.0004618, 0.0009699, -0.0009941, 0.0009675
1: -0.0008403, 0.0027763, -0.0008417, 0.0028080, -0.0027212, 0.0026983
2: 0.0121821, 0.0175984, 0.0121346, 0.0176005, -0.0037179, 0.0037647
3: -0.0014665, 0.0026064, -0.0015022, 0.0026079, -0.0026477, 0.0026894
4: -0.0057323, -0.0019755, -0.0057652, -0.0019741, -0.0035461, 0.0035476
5: 0.0064744, 0.0105398, 0.0064387, 0.0105414, -0.0026287, 0.0026710
6: 0.0076678, 0.0104896, 0.0076197, 0.0105031, -0.0028353, 0.0028700
7: -0.0212801, -0.0124547, -0.0212835, -0.0123773, -0.0042539, 0.0040949
8: 0.9628208, 0.9881068, 0.9628110, 0.9883285, -0.0178974, 0.0176912
9: 0.0011486, 0.0085801, 0.0010834, 0.0085830, -0.0039330, 0.0040402

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A1_A1_B1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114373, upper bound: 0.0114425
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114425, upper bound: 0.0114425
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004665, 0.0009445, -0.0004720, 0.0009310, -0.0009544, 0.0009905
1: -0.0008635, 0.0027691, -0.0008894, 0.0027484, -0.0026921, 0.0026994
2: 0.0121930, 0.0176331, 0.0122239, 0.0176720, -0.0037434, 0.0037181
3: -0.0014583, 0.0026325, -0.0014350, 0.0026617, -0.0026782, 0.0026524
4: -0.0057248, -0.0019515, -0.0057033, -0.0019245, -0.0035041, 0.0035197
5: 0.0064825, 0.0105658, 0.0065058, 0.0105950, -0.0026602, 0.0026338
6: 0.0076788, 0.0104866, 0.0077101, 0.0104778, -0.0027990, 0.0027764
7: -0.0213367, -0.0124724, -0.0214001, -0.0125228, -0.0041594, 0.0043545
8: 0.9626588, 0.9880561, 0.9624772, 0.9879116, -0.0176831, 0.0177860
9: 0.0011635, 0.0086278, 0.0012059, 0.0086811, -0.0040818, 0.0039663

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A1_A1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113797, upper bound: 0.0113878
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_B2_A1_A2

### Relational analysis result of IS_A1_B1_A1_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113797, upper bound: 0.0113971
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004615, 0.0009492, -0.0004721, 0.0009348, -0.0009622, 0.0009887
1: -0.0008403, 0.0027763, -0.0008899, 0.0027542, -0.0027143, 0.0026973
2: 0.0121821, 0.0175984, 0.0122153, 0.0176727, -0.0037352, 0.0037544
3: -0.0014665, 0.0026064, -0.0014415, 0.0026623, -0.0026697, 0.0026817
4: -0.0057323, -0.0019755, -0.0057093, -0.0019240, -0.0035111, 0.0035405
5: 0.0064744, 0.0105398, 0.0064993, 0.0105956, -0.0026516, 0.0026633
6: 0.0076678, 0.0104896, 0.0077014, 0.0104803, -0.0028125, 0.0027883
7: -0.0212801, -0.0124547, -0.0214013, -0.0125087, -0.0042372, 0.0042897
8: 0.9628208, 0.9881068, 0.9624736, 0.9879520, -0.0178496, 0.0177516
9: 0.0011486, 0.0085801, 0.0011940, 0.0086822, -0.0040483, 0.0040261

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A1_A1_B1_B2_A2_A1

### Relational analysis result of IS_A1_B1_A1_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113867, upper bound: 0.0113730
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_B2_A2_A2

### Relational analysis result of IS_A1_B1_A1_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113867, upper bound: 0.0113787
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0004617, 0.0009662, -0.0004411, 0.0010393, -0.0010703, 0.0009895
1: -0.0008412, 0.0028024, -0.0007448, 0.0029144, -0.0029739, 0.0027493
2: 0.0121431, 0.0175998, 0.0119754, 0.0174555, -0.0037826, 0.0041365
3: -0.0014958, 0.0026074, -0.0016219, 0.0024989, -0.0026915, 0.0029662
4: -0.0057593, -0.0019746, -0.0058757, -0.0020747, -0.0036409, 0.0038233
5: 0.0064451, 0.0105408, 0.0063192, 0.0104325, -0.0026719, 0.0029469
6: 0.0076283, 0.0105007, 0.0074584, 0.0105482, -0.0029199, 0.0030423
7: -0.0212824, -0.0123911, -0.0210472, -0.0121178, -0.0048109, 0.0042460
8: 0.9628144, 0.9882889, 0.9634882, 0.9890720, -0.0196390, 0.0180042
9: 0.0010950, 0.0085820, 0.0008649, 0.0083840, -0.0040197, 0.0045244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_B1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112424, upper bound: 0.0112212
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_B1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112568, upper bound: 0.0112199
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0004618, 0.0009699, -0.0004362, 0.0010441, -0.0010705, 0.0009953
1: -0.0008417, 0.0028080, -0.0007217, 0.0029218, -0.0029720, 0.0027681
2: 0.0121346, 0.0176005, 0.0119643, 0.0174207, -0.0038094, 0.0041278
3: -0.0015022, 0.0026079, -0.0016303, 0.0024728, -0.0027130, 0.0029559
4: -0.0057652, -0.0019741, -0.0058834, -0.0020988, -0.0036605, 0.0038303
5: 0.0064387, 0.0105414, 0.0063109, 0.0104064, -0.0026937, 0.0029364
6: 0.0076197, 0.0105031, 0.0074472, 0.0105513, -0.0029317, 0.0030559
7: -0.0212835, -0.0123773, -0.0209906, -0.0120997, -0.0047627, 0.0042942
8: 0.9628110, 0.9883285, 0.9636502, 0.9891238, -0.0196047, 0.0181287
9: 0.0010834, 0.0085830, 0.0008497, 0.0083364, -0.0040679, 0.0044954

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_B2_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112298, upper bound: 0.0112272
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_B2_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112351, upper bound: 0.0112245
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0004617, 0.0009662, -0.0004518, 0.0010047, -0.0010396, 0.0010054
1: -0.0008412, 0.0028024, -0.0007948, 0.0028614, -0.0029372, 0.0027455
2: 0.0121431, 0.0175998, 0.0120548, 0.0175302, -0.0037897, 0.0040817
3: -0.0014958, 0.0026074, -0.0015622, 0.0025551, -0.0027045, 0.0029249
4: -0.0057593, -0.0019746, -0.0058206, -0.0020228, -0.0036056, 0.0037852
5: 0.0064451, 0.0105408, 0.0063788, 0.0104886, -0.0026860, 0.0029057
6: 0.0076283, 0.0105007, 0.0075388, 0.0105257, -0.0028974, 0.0029619
7: -0.0212824, -0.0123911, -0.0211691, -0.0122471, -0.0047215, 0.0043989
8: 0.9628144, 0.9882889, 0.9631390, 0.9887015, -0.0193829, 0.0180260
9: 0.0010950, 0.0085820, 0.0009738, 0.0084866, -0.0041143, 0.0044492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_B1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111413, upper bound: 0.0111267
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_B1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111464, upper bound: 0.0111238
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0004618, 0.0009699, -0.0004472, 0.0010101, -0.0010398, 0.0010120
1: -0.0008417, 0.0028080, -0.0007732, 0.0028696, -0.0029354, 0.0027653
2: 0.0121346, 0.0176005, 0.0120425, 0.0174979, -0.0038207, 0.0040729
3: -0.0015022, 0.0026079, -0.0015715, 0.0025308, -0.0027291, 0.0029147
4: -0.0057652, -0.0019741, -0.0058291, -0.0020452, -0.0036256, 0.0037923
5: 0.0064387, 0.0105414, 0.0063696, 0.0104644, -0.0027106, 0.0028952
6: 0.0076197, 0.0105031, 0.0075264, 0.0105292, -0.0029095, 0.0029767
7: -0.0212835, -0.0123773, -0.0211164, -0.0122271, -0.0046733, 0.0044536
8: 0.9628110, 0.9883285, 0.9632899, 0.9887587, -0.0193485, 0.0181691
9: 0.0010834, 0.0085830, 0.0009570, 0.0084423, -0.0041566, 0.0044201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_B2_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111436, upper bound: 0.0111418
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_B2_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111453, upper bound: 0.0111405
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004411, 0.0010393, -0.0004617, 0.0009662, -0.0009895, 0.0010703
1: -0.0007448, 0.0029144, -0.0008412, 0.0028024, -0.0027493, 0.0029739
2: 0.0119754, 0.0174555, 0.0121431, 0.0175998, -0.0041365, 0.0037826
3: -0.0016219, 0.0024989, -0.0014958, 0.0026074, -0.0029662, 0.0026915
4: -0.0058757, -0.0020747, -0.0057593, -0.0019746, -0.0038233, 0.0036409
5: 0.0063192, 0.0104325, 0.0064451, 0.0105408, -0.0029469, 0.0026719
6: 0.0074584, 0.0105482, 0.0076283, 0.0105007, -0.0030423, 0.0029199
7: -0.0210472, -0.0121178, -0.0212824, -0.0123911, -0.0042460, 0.0048109
8: 0.9634882, 0.9890720, 0.9628144, 0.9882889, -0.0180042, 0.0196390
9: 0.0008649, 0.0083840, 0.0010950, 0.0085820, -0.0045244, 0.0040197

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112211, upper bound: 0.0112353
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112195, upper bound: 0.0112429
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004411, 0.0010393, -0.0004364, 0.0010600, -0.0010568, 0.0010390
1: -0.0007448, 0.0029144, -0.0007225, 0.0029462, -0.0028927, 0.0028910
2: 0.0119754, 0.0174555, 0.0119277, 0.0174221, -0.0039789, 0.0039852
3: -0.0016219, 0.0024989, -0.0016578, 0.0024738, -0.0028305, 0.0028365
4: -0.0058757, -0.0020747, -0.0059088, -0.0020978, -0.0037779, 0.0037999
5: 0.0063192, 0.0104325, 0.0062834, 0.0104074, -0.0028101, 0.0028161
6: 0.0074584, 0.0105482, 0.0074101, 0.0105617, -0.0031033, 0.0031381
7: -0.0210472, -0.0121178, -0.0209928, -0.0120401, -0.0043391, 0.0043112
8: 0.9634882, 0.9890720, 0.9636441, 0.9892946, -0.0189669, 0.0189406
9: 0.0008649, 0.0083840, 0.0007995, 0.0083382, -0.0041839, 0.0042015

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A1_A2_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112216, upper bound: 0.0112448
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112195, upper bound: 0.0112429
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004362, 0.0010441, -0.0004618, 0.0009699, -0.0009953, 0.0010705
1: -0.0007217, 0.0029218, -0.0008417, 0.0028080, -0.0027681, 0.0029720
2: 0.0119643, 0.0174207, 0.0121346, 0.0176005, -0.0041278, 0.0038094
3: -0.0016303, 0.0024728, -0.0015022, 0.0026079, -0.0029559, 0.0027130
4: -0.0058834, -0.0020988, -0.0057652, -0.0019741, -0.0038303, 0.0036605
5: 0.0063109, 0.0104064, 0.0064387, 0.0105414, -0.0029364, 0.0026937
6: 0.0074472, 0.0105513, 0.0076197, 0.0105031, -0.0030559, 0.0029317
7: -0.0209906, -0.0120997, -0.0212835, -0.0123773, -0.0042942, 0.0047627
8: 0.9636502, 0.9891238, 0.9628110, 0.9883285, -0.0181287, 0.0196047
9: 0.0008497, 0.0083364, 0.0010834, 0.0085830, -0.0044954, 0.0040679

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112272, upper bound: 0.0112225
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112245, upper bound: 0.0112232
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004362, 0.0010441, -0.0004365, 0.0010638, -0.0010647, 0.0010390
1: -0.0007217, 0.0029218, -0.0007231, 0.0029519, -0.0029098, 0.0028911
2: 0.0119643, 0.0174207, 0.0119192, 0.0174229, -0.0039742, 0.0040125
3: -0.0016303, 0.0024728, -0.0016642, 0.0024744, -0.0028237, 0.0028585
4: -0.0058834, -0.0020988, -0.0059147, -0.0020972, -0.0037862, 0.0038159
5: 0.0063109, 0.0104064, 0.0062770, 0.0104081, -0.0028028, 0.0028383
6: 0.0074472, 0.0105513, 0.0074015, 0.0105641, -0.0031170, 0.0031499
7: -0.0209906, -0.0120997, -0.0209942, -0.0120262, -0.0044112, 0.0042598
8: 0.9636502, 0.9891238, 0.9636400, 0.9893345, -0.0190911, 0.0189247
9: 0.0008497, 0.0083364, 0.0007877, 0.0083394, -0.0041542, 0.0042539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A1_A2_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112278, upper bound: 0.0112261
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112245, upper bound: 0.0112232
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004411, 0.0010393, -0.0004720, 0.0009310, -0.0009574, 0.0010924
1: -0.0007448, 0.0029144, -0.0008894, 0.0027484, -0.0027399, 0.0029730
2: 0.0119754, 0.0174555, 0.0122239, 0.0176720, -0.0041530, 0.0037686
3: -0.0016219, 0.0024989, -0.0014350, 0.0026617, -0.0029862, 0.0026810
4: -0.0058757, -0.0020747, -0.0057033, -0.0019245, -0.0037882, 0.0036286
5: 0.0063192, 0.0104325, 0.0065058, 0.0105950, -0.0029677, 0.0026614
6: 0.0074584, 0.0105482, 0.0077101, 0.0104778, -0.0030194, 0.0028381
7: -0.0210472, -0.0121178, -0.0214001, -0.0125228, -0.0042232, 0.0050220
8: 0.9634882, 0.9890720, 0.9624772, 0.9879116, -0.0179388, 0.0196984
9: 0.0008649, 0.0083840, 0.0012059, 0.0086811, -0.0046438, 0.0040005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111225, upper bound: 0.0111339
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111164, upper bound: 0.0111383
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004411, 0.0010393, -0.0004474, 0.0010254, -0.0010274, 0.0010598
1: -0.0007448, 0.0029144, -0.0007741, 0.0028932, -0.0028797, 0.0028896
2: 0.0119754, 0.0174555, 0.0120071, 0.0174993, -0.0039928, 0.0039657
3: -0.0016219, 0.0024989, -0.0015980, 0.0025318, -0.0028478, 0.0028219
4: -0.0058757, -0.0020747, -0.0058537, -0.0020442, -0.0037751, 0.0037790
5: 0.0063192, 0.0104325, 0.0063430, 0.0104654, -0.0028279, 0.0028015
6: 0.0074584, 0.0105482, 0.0074906, 0.0105392, -0.0030808, 0.0030576
7: -0.0210472, -0.0121178, -0.0211187, -0.0121696, -0.0043074, 0.0045110
8: 0.9634882, 0.9890720, 0.9632834, 0.9889237, -0.0188760, 0.0189893
9: 0.0008649, 0.0083840, 0.0009085, 0.0084442, -0.0042799, 0.0041748

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111225, upper bound: 0.0111339
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111164, upper bound: 0.0111383
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004362, 0.0010441, -0.0004721, 0.0009348, -0.0009634, 0.0010917
1: -0.0007217, 0.0029218, -0.0008899, 0.0027542, -0.0027613, 0.0029710
2: 0.0119643, 0.0174207, 0.0122153, 0.0176727, -0.0041451, 0.0037991
3: -0.0016303, 0.0024728, -0.0014415, 0.0026623, -0.0029779, 0.0027053
4: -0.0058834, -0.0020988, -0.0057093, -0.0019240, -0.0037954, 0.0036105
5: 0.0063109, 0.0104064, 0.0064993, 0.0105956, -0.0029592, 0.0026861
6: 0.0074472, 0.0105513, 0.0077014, 0.0104803, -0.0030331, 0.0028500
7: -0.0209906, -0.0120997, -0.0214013, -0.0125087, -0.0042775, 0.0049576
8: 0.9636502, 0.9891238, 0.9624736, 0.9879520, -0.0180808, 0.0196651
9: 0.0008497, 0.0083364, 0.0011940, 0.0086822, -0.0046107, 0.0040538

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111354, upper bound: 0.0111342
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111303, upper bound: 0.0111359
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004362, 0.0010441, -0.0004475, 0.0010293, -0.0010357, 0.0010590
1: -0.0007217, 0.0029218, -0.0007746, 0.0028991, -0.0029004, 0.0028886
2: 0.0119643, 0.0174207, 0.0119983, 0.0175001, -0.0039858, 0.0039984
3: -0.0016303, 0.0024728, -0.0016047, 0.0025324, -0.0028404, 0.0028479
4: -0.0058834, -0.0020988, -0.0058598, -0.0020437, -0.0037824, 0.0037611
5: 0.0063109, 0.0104064, 0.0063364, 0.0104660, -0.0028202, 0.0028277
6: 0.0074472, 0.0105513, 0.0074816, 0.0105417, -0.0030945, 0.0030697
7: -0.0209906, -0.0120997, -0.0211199, -0.0121551, -0.0043883, 0.0044478
8: 0.9636502, 0.9891238, 0.9632798, 0.9889650, -0.0190255, 0.0189628
9: 0.0008497, 0.0083364, 0.0008963, 0.0084452, -0.0042557, 0.0042346

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111354, upper bound: 0.0111342
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111303, upper bound: 0.0111359
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0004720, 0.0009310, -0.0004665, 0.0009445, -0.0009905, 0.0009544
1: -0.0008894, 0.0027484, -0.0008635, 0.0027691, -0.0026994, 0.0026921
2: 0.0122239, 0.0176720, 0.0121930, 0.0176331, -0.0037181, 0.0037434
3: -0.0014350, 0.0026617, -0.0014583, 0.0026325, -0.0026524, 0.0026782
4: -0.0057033, -0.0019245, -0.0057248, -0.0019515, -0.0035197, 0.0035041
5: 0.0065058, 0.0105950, 0.0064825, 0.0105658, -0.0026338, 0.0026602
6: 0.0077101, 0.0104778, 0.0076788, 0.0104866, -0.0027764, 0.0027990
7: -0.0214001, -0.0125228, -0.0213367, -0.0124724, -0.0043545, 0.0041594
8: 0.9624772, 0.9879116, 0.9626588, 0.9880561, -0.0177860, 0.0176831
9: 0.0012059, 0.0086811, 0.0011635, 0.0086278, -0.0039663, 0.0040818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113870, upper bound: 0.0113797
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113951, upper bound: 0.0113797
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0004720, 0.0009310, -0.0004765, 0.0009092, -0.0009424, 0.0009607
1: -0.0008894, 0.0027484, -0.0009105, 0.0027150, -0.0026371, 0.0026413
2: 0.0122239, 0.0176720, 0.0122739, 0.0177036, -0.0036571, 0.0036476
3: -0.0014350, 0.0026617, -0.0013974, 0.0026855, -0.0026130, 0.0026043
4: -0.0057033, -0.0019245, -0.0056686, -0.0019026, -0.0034380, 0.0034473
5: 0.0065058, 0.0105950, 0.0065433, 0.0106187, -0.0025951, 0.0025863
6: 0.0077101, 0.0104778, 0.0077607, 0.0104636, -0.0027535, 0.0027171
7: -0.0214001, -0.0125228, -0.0214515, -0.0126043, -0.0040982, 0.0041508
8: 0.9624772, 0.9879116, 0.9623297, 0.9876783, -0.0173422, 0.0173838
9: 0.0012059, 0.0086811, 0.0012745, 0.0087245, -0.0039258, 0.0038970

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113951, upper bound: 0.0113706
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B2_A2

### Relational analysis result of IS_A1_B1_A2_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113951, upper bound: 0.0113797
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0004721, 0.0009348, -0.0004615, 0.0009492, -0.0009887, 0.0009622
1: -0.0008899, 0.0027542, -0.0008403, 0.0027763, -0.0026973, 0.0027143
2: 0.0122153, 0.0176727, 0.0121821, 0.0175984, -0.0037544, 0.0037352
3: -0.0014415, 0.0026623, -0.0014665, 0.0026064, -0.0026817, 0.0026697
4: -0.0057093, -0.0019240, -0.0057323, -0.0019755, -0.0035405, 0.0035111
5: 0.0064993, 0.0105956, 0.0064744, 0.0105398, -0.0026633, 0.0026516
6: 0.0077014, 0.0104803, 0.0076678, 0.0104896, -0.0027883, 0.0028125
7: -0.0214013, -0.0125087, -0.0212801, -0.0124547, -0.0042897, 0.0042372
8: 0.9624736, 0.9879520, 0.9628208, 0.9881068, -0.0177516, 0.0178496
9: 0.0011940, 0.0086822, 0.0011486, 0.0085801, -0.0040261, 0.0040483

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_B1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113718, upper bound: 0.0113867
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_B1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113779, upper bound: 0.0113867
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0004721, 0.0009348, -0.0004718, 0.0009138, -0.0009412, 0.0009683
1: -0.0008899, 0.0027542, -0.0008885, 0.0027221, -0.0026336, 0.0026586
2: 0.0122153, 0.0176727, 0.0122634, 0.0176707, -0.0036859, 0.0036367
3: -0.0014415, 0.0026623, -0.0014053, 0.0026607, -0.0026367, 0.0025938
4: -0.0057093, -0.0019240, -0.0056759, -0.0019254, -0.0034569, 0.0034493
5: 0.0064993, 0.0105956, 0.0065354, 0.0105940, -0.0026189, 0.0025755
6: 0.0077014, 0.0104803, 0.0077501, 0.0104666, -0.0027653, 0.0027302
7: -0.0214013, -0.0125087, -0.0213979, -0.0125871, -0.0040545, 0.0042108
8: 0.9624736, 0.9879520, 0.9624835, 0.9877274, -0.0172972, 0.0175135
9: 0.0011940, 0.0086822, 0.0012601, 0.0086793, -0.0039765, 0.0038658

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_B2_A1

### Relational analysis result of IS_A1_B1_A2_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113779, upper bound: 0.0113794
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_B2_A2

### Relational analysis result of IS_A1_B1_A2_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113779, upper bound: 0.0113867
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0004720, 0.0009310, -0.0004411, 0.0010393, -0.0010924, 0.0009574
1: -0.0008894, 0.0027484, -0.0007448, 0.0029144, -0.0029730, 0.0027399
2: 0.0122239, 0.0176720, 0.0119754, 0.0174555, -0.0037686, 0.0041530
3: -0.0014350, 0.0026617, -0.0016219, 0.0024989, -0.0026810, 0.0029862
4: -0.0057033, -0.0019245, -0.0058757, -0.0020747, -0.0036286, 0.0037882
5: 0.0065058, 0.0105950, 0.0063192, 0.0104325, -0.0026614, 0.0029677
6: 0.0077101, 0.0104778, 0.0074584, 0.0105482, -0.0028381, 0.0030194
7: -0.0214001, -0.0125228, -0.0210472, -0.0121178, -0.0050220, 0.0042232
8: 0.9624772, 0.9879116, 0.9634882, 0.9890720, -0.0196984, 0.0179388
9: 0.0012059, 0.0086811, 0.0008649, 0.0083840, -0.0040005, 0.0046438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111344, upper bound: 0.0111278
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111429, upper bound: 0.0111231
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0004720, 0.0009310, -0.0004518, 0.0010047, -0.0010465, 0.0009637
1: -0.0008894, 0.0027484, -0.0007948, 0.0028614, -0.0029092, 0.0026887
2: 0.0122239, 0.0176720, 0.0120548, 0.0175302, -0.0037051, 0.0040552
3: -0.0014350, 0.0026617, -0.0015622, 0.0025551, -0.0026385, 0.0029108
4: -0.0057033, -0.0019245, -0.0058206, -0.0020228, -0.0035494, 0.0037300
5: 0.0065058, 0.0105950, 0.0063788, 0.0104886, -0.0026195, 0.0028923
6: 0.0077101, 0.0104778, 0.0075388, 0.0105257, -0.0028156, 0.0029390
7: -0.0214001, -0.0125228, -0.0211691, -0.0122471, -0.0047624, 0.0042159
8: 0.9624772, 0.9879116, 0.9631390, 0.9887015, -0.0192451, 0.0176315
9: 0.0012059, 0.0086811, 0.0009738, 0.0084866, -0.0039644, 0.0044563

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B2_B1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111344, upper bound: 0.0111278
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B2_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111429, upper bound: 0.0111231
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0004721, 0.0009348, -0.0004362, 0.0010441, -0.0010917, 0.0009634
1: -0.0008899, 0.0027542, -0.0007217, 0.0029218, -0.0029710, 0.0027613
2: 0.0122153, 0.0176727, 0.0119643, 0.0174207, -0.0037991, 0.0041451
3: -0.0014415, 0.0026623, -0.0016303, 0.0024728, -0.0027053, 0.0029779
4: -0.0057093, -0.0019240, -0.0058834, -0.0020988, -0.0036105, 0.0037954
5: 0.0064993, 0.0105956, 0.0063109, 0.0104064, -0.0026861, 0.0029592
6: 0.0077014, 0.0104803, 0.0074472, 0.0105513, -0.0028500, 0.0030331
7: -0.0214013, -0.0125087, -0.0209906, -0.0120997, -0.0049576, 0.0042775
8: 0.9624736, 0.9879520, 0.9636502, 0.9891238, -0.0196651, 0.0180808
9: 0.0011940, 0.0086822, 0.0008497, 0.0083364, -0.0040538, 0.0046107

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111375, upper bound: 0.0111434
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111407, upper bound: 0.0111392
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0004721, 0.0009348, -0.0004472, 0.0010101, -0.0010463, 0.0009698
1: -0.0008899, 0.0027542, -0.0007732, 0.0028696, -0.0029064, 0.0027080
2: 0.0122153, 0.0176727, 0.0120425, 0.0174979, -0.0037354, 0.0040453
3: -0.0014415, 0.0026623, -0.0015715, 0.0025308, -0.0026632, 0.0029010
4: -0.0057093, -0.0019240, -0.0058291, -0.0020452, -0.0035687, 0.0037327
5: 0.0064993, 0.0105956, 0.0063696, 0.0104644, -0.0026445, 0.0028822
6: 0.0077014, 0.0104803, 0.0075264, 0.0105292, -0.0028278, 0.0029539
7: -0.0214013, -0.0125087, -0.0211164, -0.0122271, -0.0047202, 0.0042634
8: 0.9624736, 0.9879520, 0.9632899, 0.9887587, -0.0192045, 0.0177710
9: 0.0011940, 0.0086822, 0.0009570, 0.0084423, -0.0040136, 0.0044263

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B2_B1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111375, upper bound: 0.0111434
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B2_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111407, upper bound: 0.0111392
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A2_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0004518, 0.0010047, -0.0004617, 0.0009662, -0.0010054, 0.0010396
1: -0.0007948, 0.0028614, -0.0008412, 0.0028024, -0.0027455, 0.0029372
2: 0.0120548, 0.0175302, 0.0121431, 0.0175998, -0.0040817, 0.0037897
3: -0.0015622, 0.0025551, -0.0014958, 0.0026074, -0.0029249, 0.0027045
4: -0.0058206, -0.0020228, -0.0057593, -0.0019746, -0.0037852, 0.0036056
5: 0.0063788, 0.0104886, 0.0064451, 0.0105408, -0.0029057, 0.0026860
6: 0.0075388, 0.0105257, 0.0076283, 0.0105007, -0.0029619, 0.0028974
7: -0.0211691, -0.0122471, -0.0212824, -0.0123911, -0.0043989, 0.0047215
8: 0.9631390, 0.9887015, 0.9628144, 0.9882889, -0.0180260, 0.0193829
9: 0.0009738, 0.0084866, 0.0010950, 0.0085820, -0.0044492, 0.0041143

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A2_A2_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_A2_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111248, upper bound: 0.0111411
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_A2_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_A2_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111213, upper bound: 0.0111456
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0004518, 0.0010047, -0.0004720, 0.0009310, -0.0009637, 0.0010465
1: -0.0007948, 0.0028614, -0.0008894, 0.0027484, -0.0026887, 0.0029092
2: 0.0120548, 0.0175302, 0.0122239, 0.0176720, -0.0040552, 0.0037051
3: -0.0015622, 0.0025551, -0.0014350, 0.0026617, -0.0029108, 0.0026385
4: -0.0058206, -0.0020228, -0.0057033, -0.0019245, -0.0037300, 0.0035494
5: 0.0063788, 0.0104886, 0.0065058, 0.0105950, -0.0028923, 0.0026195
6: 0.0075388, 0.0105257, 0.0077101, 0.0104778, -0.0029390, 0.0028156
7: -0.0211691, -0.0122471, -0.0214001, -0.0125228, -0.0042159, 0.0047624
8: 0.9631390, 0.9887015, 0.9624772, 0.9879116, -0.0176315, 0.0192451
9: 0.0009738, 0.0084866, 0.0012059, 0.0086811, -0.0044563, 0.0039644

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A2_A2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111248, upper bound: 0.0111411
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_A2_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_A2_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111213, upper bound: 0.0111456
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0004518, 0.0010047, -0.0004364, 0.0010600, -0.0010768, 0.0010089
1: -0.0007948, 0.0028614, -0.0007225, 0.0029462, -0.0028913, 0.0028739
2: 0.0120548, 0.0175302, 0.0119277, 0.0174221, -0.0039534, 0.0039981
3: -0.0015622, 0.0025551, -0.0016578, 0.0024738, -0.0028113, 0.0028539
4: -0.0058206, -0.0020228, -0.0059088, -0.0020978, -0.0037228, 0.0037658
5: 0.0063788, 0.0104886, 0.0062834, 0.0104074, -0.0027909, 0.0028343
6: 0.0075388, 0.0105257, 0.0074101, 0.0105617, -0.0030229, 0.0031156
7: -0.0211691, -0.0122471, -0.0209928, -0.0120401, -0.0045275, 0.0042696
8: 0.9631390, 0.9887015, 0.9636441, 0.9892946, -0.0190113, 0.0188214
9: 0.0009738, 0.0084866, 0.0007995, 0.0083382, -0.0041488, 0.0043056

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A2_A2_A1_B2_B1_B1

### Relational analysis result of IS_A1_B1_A2_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111129, upper bound: 0.0111277
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_A2_A1_B2_B1_B2

### Relational analysis result of IS_A1_B1_A2_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111116, upper bound: 0.0111236
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0004518, 0.0010047, -0.0004474, 0.0010254, -0.0010333, 0.0010146
1: -0.0007948, 0.0028614, -0.0007741, 0.0028932, -0.0028289, 0.0028262
2: 0.0120548, 0.0175302, 0.0120071, 0.0174993, -0.0038930, 0.0039020
3: -0.0015622, 0.0025551, -0.0015980, 0.0025318, -0.0027729, 0.0027807
4: -0.0058206, -0.0020228, -0.0058537, -0.0020442, -0.0037164, 0.0037059
5: 0.0063788, 0.0104886, 0.0063430, 0.0104654, -0.0027531, 0.0027609
6: 0.0075388, 0.0105257, 0.0074906, 0.0105392, -0.0030004, 0.0030352
7: -0.0211691, -0.0122471, -0.0211187, -0.0121696, -0.0042972, 0.0042574
8: 0.9631390, 0.9887015, 0.9632834, 0.9889237, -0.0185653, 0.0185272
9: 0.0009738, 0.0084866, 0.0009085, 0.0084442, -0.0041115, 0.0041353

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A2_A2_A1_B2_B2_B1

### Relational analysis result of IS_A1_B1_A2_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111129, upper bound: 0.0111277
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A2_A2_A1_B2_B2_B2

### Relational analysis result of IS_A1_B1_A2_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111116, upper bound: 0.0111236
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0004472, 0.0010101, -0.0004618, 0.0009699, -0.0010120, 0.0010398
1: -0.0007732, 0.0028696, -0.0008417, 0.0028080, -0.0027653, 0.0029354
2: 0.0120425, 0.0174979, 0.0121346, 0.0176005, -0.0040729, 0.0038207
3: -0.0015715, 0.0025308, -0.0015022, 0.0026079, -0.0029147, 0.0027291
4: -0.0058291, -0.0020452, -0.0057652, -0.0019741, -0.0037923, 0.0036256
5: 0.0063696, 0.0104644, 0.0064387, 0.0105414, -0.0028952, 0.0027106
6: 0.0075264, 0.0105292, 0.0076197, 0.0105031, -0.0029767, 0.0029095
7: -0.0211164, -0.0122271, -0.0212835, -0.0123773, -0.0044536, 0.0046733
8: 0.9632899, 0.9887587, 0.9628110, 0.9883285, -0.0181691, 0.0193485
9: 0.0009570, 0.0084423, 0.0010834, 0.0085830, -0.0044201, 0.0041566

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A2_A2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_A2_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111378, upper bound: 0.0111436
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_A2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B1_A2_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111359, upper bound: 0.0111452
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A2_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0004472, 0.0010101, -0.0004721, 0.0009348, -0.0009698, 0.0010463
1: -0.0007732, 0.0028696, -0.0008899, 0.0027542, -0.0027080, 0.0029064
2: 0.0120425, 0.0174979, 0.0122153, 0.0176727, -0.0040453, 0.0037354
3: -0.0015715, 0.0025308, -0.0014415, 0.0026623, -0.0029010, 0.0026632
4: -0.0058291, -0.0020452, -0.0057093, -0.0019240, -0.0037327, 0.0035687
5: 0.0063696, 0.0104644, 0.0064993, 0.0105956, -0.0028822, 0.0026445
6: 0.0075264, 0.0105292, 0.0077014, 0.0104803, -0.0029539, 0.0028278
7: -0.0211164, -0.0122271, -0.0214013, -0.0125087, -0.0042634, 0.0047202
8: 0.9632899, 0.9887587, 0.9624736, 0.9879520, -0.0177710, 0.0192045
9: 0.0009570, 0.0084423, 0.0011940, 0.0086822, -0.0044263, 0.0040136

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A2_A2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111378, upper bound: 0.0111436
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A2_A2_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_A2_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111359, upper bound: 0.0111452
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0004472, 0.0010101, -0.0004365, 0.0010638, -0.0010855, 0.0010098
1: -0.0007732, 0.0028696, -0.0007231, 0.0029519, -0.0029087, 0.0028740
2: 0.0120425, 0.0174979, 0.0119192, 0.0174229, -0.0039485, 0.0040273
3: -0.0015715, 0.0025308, -0.0016642, 0.0024744, -0.0028044, 0.0028768
4: -0.0058291, -0.0020452, -0.0059147, -0.0020972, -0.0037319, 0.0037830
5: 0.0063696, 0.0104644, 0.0062770, 0.0104081, -0.0027835, 0.0028571
6: 0.0075264, 0.0105292, 0.0074015, 0.0105641, -0.0030378, 0.0031277
7: -0.0211164, -0.0122271, -0.0209942, -0.0120262, -0.0046114, 0.0042179
8: 0.9632899, 0.9887587, 0.9636400, 0.9893345, -0.0191438, 0.0188048
9: 0.0009570, 0.0084423, 0.0007877, 0.0083394, -0.0041190, 0.0043507

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A2_A2_A2_B2_B1_B1

### Relational analysis result of IS_A1_B1_A2_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111304, upper bound: 0.0111332
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_A2_A2_B2_B1_B2

### Relational analysis result of IS_A1_B1_A2_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111266, upper bound: 0.0111287
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0004472, 0.0010101, -0.0004475, 0.0010293, -0.0010411, 0.0010156
1: -0.0007732, 0.0028696, -0.0007746, 0.0028991, -0.0028465, 0.0028252
2: 0.0120425, 0.0174979, 0.0119983, 0.0175001, -0.0038882, 0.0039291
3: -0.0015715, 0.0025308, -0.0016047, 0.0025324, -0.0027659, 0.0028024
4: -0.0058291, -0.0020452, -0.0058598, -0.0020437, -0.0037205, 0.0037258
5: 0.0063696, 0.0104644, 0.0063364, 0.0104660, -0.0027458, 0.0027828
6: 0.0075264, 0.0105292, 0.0074816, 0.0105417, -0.0030153, 0.0030476
7: -0.0211164, -0.0122271, -0.0211199, -0.0121551, -0.0043602, 0.0042136
8: 0.9632899, 0.9887587, 0.9632798, 0.9889650, -0.0186907, 0.0185094
9: 0.0009570, 0.0084423, 0.0008963, 0.0084452, -0.0040851, 0.0041838

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A2_A2_A2_B2_B2_B1

### Relational analysis result of IS_A1_B1_A2_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111304, upper bound: 0.0111332
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_A2_A2_B2_B2_B2

### Relational analysis result of IS_A1_B1_A2_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111266, upper bound: 0.0111287
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0004665, 0.0009445, -0.0004867, 0.0011974, -0.0012345, 0.0010630
1: -0.0008635, 0.0027691, -0.0009580, 0.0031566, -0.0031888, 0.0027789
2: 0.0121930, 0.0176331, 0.0116126, 0.0177747, -0.0038601, 0.0044619
3: -0.0014583, 0.0026325, -0.0018947, 0.0027389, -0.0027756, 0.0032117
4: -0.0057248, -0.0019515, -0.0061273, -0.0018532, -0.0036869, 0.0040356
5: 0.0064825, 0.0105658, 0.0060469, 0.0106721, -0.0027588, 0.0031921
6: 0.0076788, 0.0104866, 0.0070910, 0.0106510, -0.0029722, 0.0033955
7: -0.0213367, -0.0124724, -0.0215674, -0.0115267, -0.0053713, 0.0050763
8: 0.9626588, 0.9880561, 0.9619977, 0.9907656, -0.0211553, 0.0183185
9: 0.0011635, 0.0086278, 0.0003671, 0.0088220, -0.0044620, 0.0049868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111575, upper bound: 0.0120406
time: 0.76 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111610, upper bound: 0.0120406
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0004615, 0.0009492, -0.0004868, 0.0012019, -0.0012436, 0.0010659
1: -0.0008403, 0.0027763, -0.0009588, 0.0031636, -0.0032101, 0.0027873
2: 0.0121821, 0.0175984, 0.0116022, 0.0177759, -0.0038712, 0.0044970
3: -0.0014665, 0.0026064, -0.0019025, 0.0027398, -0.0027835, 0.0032401
4: -0.0057323, -0.0019755, -0.0061345, -0.0018524, -0.0036951, 0.0040555
5: 0.0064744, 0.0105398, 0.0060391, 0.0106730, -0.0027665, 0.0032207
6: 0.0076678, 0.0104896, 0.0070805, 0.0106539, -0.0029861, 0.0034091
7: -0.0212801, -0.0124547, -0.0215693, -0.0115098, -0.0054472, 0.0050849
8: 0.9628208, 0.9881068, 0.9619921, 0.9908141, -0.0213163, 0.0183717
9: 0.0011486, 0.0085801, 0.0003529, 0.0088237, -0.0044690, 0.0050450

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111662, upper bound: 0.0120342
time: 0.87 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111739, upper bound: 0.0120342
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004720, 0.0009310, -0.0004899, 0.0011744, -0.0012363, 0.0010493
1: -0.0008894, 0.0027484, -0.0009734, 0.0031214, -0.0031820, 0.0027403
2: 0.0122239, 0.0176720, 0.0116653, 0.0177977, -0.0038126, 0.0044660
3: -0.0014350, 0.0026617, -0.0018551, 0.0027562, -0.0027429, 0.0032216
4: -0.0057033, -0.0019245, -0.0060908, -0.0018373, -0.0036237, 0.0040053
5: 0.0065058, 0.0105950, 0.0060865, 0.0106894, -0.0027265, 0.0032026
6: 0.0077101, 0.0104778, 0.0071444, 0.0106360, -0.0029259, 0.0033334
7: -0.0214001, -0.0125228, -0.0216049, -0.0116125, -0.0055319, 0.0050436
8: 0.9624772, 0.9879116, 0.9618902, 0.9905197, -0.0211594, 0.0180879
9: 0.0012059, 0.0086811, 0.0004394, 0.0088536, -0.0044278, 0.0050732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111360, upper bound: 0.0119749
time: 0.79 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111434, upper bound: 0.0119748
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004721, 0.0009348, -0.0004864, 0.0011819, -0.0012376, 0.0010544
1: -0.0008899, 0.0027542, -0.0009569, 0.0031329, -0.0031794, 0.0027675
2: 0.0122153, 0.0176727, 0.0116482, 0.0177731, -0.0038485, 0.0044571
3: -0.0014415, 0.0026623, -0.0018680, 0.0027377, -0.0027685, 0.0032126
4: -0.0057093, -0.0019240, -0.0061026, -0.0018544, -0.0036607, 0.0040118
5: 0.0064993, 0.0105956, 0.0060736, 0.0106709, -0.0027519, 0.0031934
6: 0.0077014, 0.0104803, 0.0071271, 0.0106409, -0.0029395, 0.0033532
7: -0.0214013, -0.0125087, -0.0215647, -0.0115847, -0.0054661, 0.0050792
8: 0.9624736, 0.9879520, 0.9620054, 0.9905995, -0.0211219, 0.0182588
9: 0.0011940, 0.0086822, 0.0004159, 0.0088198, -0.0044609, 0.0050388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111066, upper bound: 0.0119749
time: 0.81 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B2_B1_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111127, upper bound: 0.0119748
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004617, 0.0009662, -0.0004671, 0.0012760, -0.0013286, 0.0010751
1: -0.0008412, 0.0028024, -0.0008664, 0.0032772, -0.0034143, 0.0028368
2: 0.0121431, 0.0175998, 0.0114321, 0.0176375, -0.0039223, 0.0047961
3: -0.0014958, 0.0026074, -0.0020305, 0.0026358, -0.0028103, 0.0034622
4: -0.0057593, -0.0019746, -0.0062525, -0.0019484, -0.0038049, 0.0042780
5: 0.0064451, 0.0105408, 0.0059114, 0.0105691, -0.0027927, 0.0034420
6: 0.0076283, 0.0105007, 0.0069082, 0.0107021, -0.0030738, 0.0035925
7: -0.0212824, -0.0123911, -0.0213438, -0.0112325, -0.0058856, 0.0050768
8: 0.9628144, 0.9882889, 0.9626384, 0.9916085, -0.0227183, 0.0186335
9: 0.0010950, 0.0085820, 0.0001194, 0.0086338, -0.0044952, 0.0054294

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110283, upper bound: 0.0118445
time: 0.66 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110344, upper bound: 0.0118383
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004618, 0.0009699, -0.0004648, 0.0012888, -0.0013333, 0.0010795
1: -0.0008417, 0.0028080, -0.0008558, 0.0032969, -0.0034123, 0.0028649
2: 0.0121346, 0.0176005, 0.0114026, 0.0176216, -0.0039536, 0.0047871
3: -0.0015022, 0.0026079, -0.0020526, 0.0026238, -0.0028335, 0.0034518
4: -0.0057652, -0.0019741, -0.0062730, -0.0019595, -0.0038058, 0.0042877
5: 0.0064387, 0.0105414, 0.0058893, 0.0105572, -0.0028157, 0.0034313
6: 0.0076197, 0.0105031, 0.0068784, 0.0107104, -0.0030908, 0.0036247
7: -0.0212835, -0.0123773, -0.0213179, -0.0111845, -0.0058372, 0.0050935
8: 0.9628110, 0.9883285, 0.9627127, 0.9917461, -0.0226831, 0.0187864
9: 0.0010834, 0.0085830, 0.0000790, 0.0086119, -0.0045244, 0.0054001

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109973, upper bound: 0.0118445
time: 0.79 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110035, upper bound: 0.0118383
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004720, 0.0009310, -0.0004671, 0.0012760, -0.0013507, 0.0010412
1: -0.0008894, 0.0027484, -0.0008664, 0.0032772, -0.0034134, 0.0027914
2: 0.0122239, 0.0176720, 0.0114321, 0.0176375, -0.0038543, 0.0048126
3: -0.0014350, 0.0026617, -0.0020305, 0.0026358, -0.0027592, 0.0034822
4: -0.0057033, -0.0019245, -0.0062525, -0.0019484, -0.0037549, 0.0042457
5: 0.0065058, 0.0105950, 0.0059114, 0.0105691, -0.0027416, 0.0034628
6: 0.0077101, 0.0104778, 0.0069082, 0.0107021, -0.0029920, 0.0035696
7: -0.0214001, -0.0125228, -0.0213438, -0.0112325, -0.0060968, 0.0049661
8: 0.9624772, 0.9879116, 0.9626384, 0.9916085, -0.0227778, 0.0183161
9: 0.0012059, 0.0086811, 0.0001194, 0.0086338, -0.0044020, 0.0055488

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109336, upper bound: 0.0117247
time: 0.78 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109347, upper bound: 0.0117045
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004721, 0.0009348, -0.0004648, 0.0012888, -0.0013545, 0.0010458
1: -0.0008899, 0.0027542, -0.0008558, 0.0032969, -0.0034113, 0.0028217
2: 0.0122153, 0.0176727, 0.0114026, 0.0176216, -0.0038890, 0.0048045
3: -0.0014415, 0.0026623, -0.0020526, 0.0026238, -0.0027850, 0.0034738
4: -0.0057093, -0.0019240, -0.0062730, -0.0019595, -0.0037498, 0.0042527
5: 0.0064993, 0.0105956, 0.0058893, 0.0105572, -0.0027672, 0.0034541
6: 0.0077014, 0.0104803, 0.0068784, 0.0107104, -0.0030091, 0.0036019
7: -0.0214013, -0.0125087, -0.0213179, -0.0111845, -0.0060320, 0.0049882
8: 0.9624736, 0.9879520, 0.9627127, 0.9917461, -0.0227435, 0.0184847
9: 0.0011940, 0.0086822, 0.0000790, 0.0086119, -0.0044357, 0.0055154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_B1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109127, upper bound: 0.0117348
time: 0.84 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109159, upper bound: 0.0117181
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004411, 0.0010393, -0.0004867, 0.0011974, -0.0012375, 0.0011641
1: -0.0007448, 0.0029144, -0.0009580, 0.0031566, -0.0032366, 0.0030196
2: 0.0119754, 0.0174555, 0.0116126, 0.0177747, -0.0042206, 0.0045124
3: -0.0016219, 0.0024989, -0.0018947, 0.0027389, -0.0030467, 0.0032403
4: -0.0058757, -0.0020747, -0.0061273, -0.0018532, -0.0039369, 0.0040526
5: 0.0063192, 0.0104325, 0.0060469, 0.0106721, -0.0030294, 0.0032197
6: 0.0074584, 0.0105482, 0.0070910, 0.0106510, -0.0031925, 0.0034572
7: -0.0210472, -0.0121178, -0.0215674, -0.0115267, -0.0054351, 0.0056637
8: 0.9634882, 0.9890720, 0.9619977, 0.9907656, -0.0214111, 0.0200015
9: 0.0008649, 0.0083840, 0.0003671, 0.0088220, -0.0049566, 0.0050210

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A2_A1_A1_B1_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109590, upper bound: 0.0118019
time: 0.69 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_A1_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109577, upper bound: 0.0118050
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004411, 0.0010393, -0.0004650, 0.0013005, -0.0013167, 0.0011362
1: -0.0007448, 0.0029144, -0.0008567, 0.0033146, -0.0033774, 0.0029797
2: 0.0119754, 0.0174555, 0.0113760, 0.0176231, -0.0041090, 0.0047109
3: -0.0016219, 0.0024989, -0.0020727, 0.0026249, -0.0029431, 0.0033823
4: -0.0058757, -0.0020747, -0.0062915, -0.0019584, -0.0039173, 0.0042168
5: 0.0063192, 0.0104325, 0.0058693, 0.0105583, -0.0029240, 0.0033609
6: 0.0074584, 0.0105482, 0.0068514, 0.0107180, -0.0032596, 0.0036968
7: -0.0210472, -0.0121178, -0.0213203, -0.0111411, -0.0055217, 0.0052421
8: 0.9634882, 0.9890720, 0.9627057, 0.9918704, -0.0223552, 0.0195410
9: 0.0008649, 0.0083840, 0.0000424, 0.0086140, -0.0046438, 0.0051973

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A2_A1_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109567, upper bound: 0.0118162
time: 0.65 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_A1_B2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109577, upper bound: 0.0118050
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004362, 0.0010441, -0.0004868, 0.0012019, -0.0012448, 0.0011693
1: -0.0007217, 0.0029218, -0.0009588, 0.0031636, -0.0032571, 0.0030319
2: 0.0119643, 0.0174207, 0.0116022, 0.0177759, -0.0042376, 0.0045417
3: -0.0016303, 0.0024728, -0.0019025, 0.0027398, -0.0030590, 0.0032637
4: -0.0058834, -0.0020988, -0.0061345, -0.0018524, -0.0039492, 0.0040358
5: 0.0063109, 0.0104064, 0.0060391, 0.0106730, -0.0030415, 0.0032434
6: 0.0074472, 0.0105513, 0.0070805, 0.0106539, -0.0032067, 0.0034708
7: -0.0209906, -0.0120997, -0.0215693, -0.0115098, -0.0054875, 0.0056818
8: 0.9636502, 0.9891238, 0.9619921, 0.9908141, -0.0215475, 0.0200820
9: 0.0008497, 0.0083364, 0.0003529, 0.0088237, -0.0049716, 0.0050727

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A2_A1_A2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109659, upper bound: 0.0117987
time: 0.68 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_A2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109637, upper bound: 0.0117996
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004362, 0.0010441, -0.0004652, 0.0013065, -0.0013262, 0.0011412
1: -0.0007217, 0.0029218, -0.0008575, 0.0033239, -0.0033980, 0.0029879
2: 0.0119643, 0.0174207, 0.0113621, 0.0176242, -0.0041204, 0.0047437
3: -0.0016303, 0.0024728, -0.0020831, 0.0026257, -0.0029495, 0.0034083
4: -0.0058834, -0.0020988, -0.0063011, -0.0019577, -0.0039257, 0.0042023
5: 0.0063109, 0.0104064, 0.0058589, 0.0105591, -0.0029308, 0.0033871
6: 0.0074472, 0.0105513, 0.0068374, 0.0107219, -0.0032747, 0.0037140
7: -0.0209906, -0.0120997, -0.0213221, -0.0111185, -0.0056027, 0.0052490
8: 0.9636502, 0.9891238, 0.9627005, 0.9919351, -0.0225048, 0.0195939
9: 0.0008497, 0.0083364, 0.0000234, 0.0086155, -0.0046515, 0.0052572

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A2_A1_A2_B2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109632, upper bound: 0.0118120
time: 0.68 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_A2_B2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109637, upper bound: 0.0117996
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004518, 0.0010047, -0.0004867, 0.0011974, -0.0012534, 0.0011327
1: -0.0007948, 0.0028614, -0.0009580, 0.0031566, -0.0032328, 0.0029626
2: 0.0120548, 0.0175302, 0.0116126, 0.0177747, -0.0041353, 0.0045194
3: -0.0015622, 0.0025551, -0.0018947, 0.0027389, -0.0029825, 0.0032533
4: -0.0058206, -0.0020228, -0.0061273, -0.0018532, -0.0038777, 0.0041045
5: 0.0063788, 0.0104886, 0.0060469, 0.0106721, -0.0029653, 0.0032337
6: 0.0075388, 0.0105257, 0.0070910, 0.0106510, -0.0031122, 0.0034347
7: -0.0211691, -0.0122471, -0.0215674, -0.0115267, -0.0055880, 0.0055246
8: 0.9631390, 0.9887015, 0.9619977, 0.9907656, -0.0214329, 0.0196031
9: 0.0009738, 0.0084866, 0.0003671, 0.0088220, -0.0048395, 0.0051156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A2_A2_A1_B1_A1

### Relational analysis result of IS_A1_B2_B1_A2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108664, upper bound: 0.0116512
time: 0.69 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_A1_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108642, upper bound: 0.0116506
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004518, 0.0010047, -0.0004650, 0.0013005, -0.0013367, 0.0011057
1: -0.0007948, 0.0028614, -0.0008567, 0.0033146, -0.0033760, 0.0029258
2: 0.0120548, 0.0175302, 0.0113760, 0.0176231, -0.0040283, 0.0047239
3: -0.0015622, 0.0025551, -0.0020727, 0.0026249, -0.0028824, 0.0033997
4: -0.0058206, -0.0020228, -0.0062915, -0.0019584, -0.0038622, 0.0042686
5: 0.0063788, 0.0104886, 0.0058693, 0.0105583, -0.0028634, 0.0033791
6: 0.0075388, 0.0105257, 0.0068514, 0.0107180, -0.0031792, 0.0036743
7: -0.0211691, -0.0122471, -0.0213203, -0.0111411, -0.0057101, 0.0051106
8: 0.9631390, 0.9887015, 0.9627057, 0.9918704, -0.0223996, 0.0191643
9: 0.0009738, 0.0084866, 0.0000424, 0.0086140, -0.0045331, 0.0053014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A2_A2_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108630, upper bound: 0.0116720
time: 0.81 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_A1_B2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108642, upper bound: 0.0116506
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004472, 0.0010101, -0.0004868, 0.0012019, -0.0012616, 0.0011379
1: -0.0007732, 0.0028696, -0.0009588, 0.0031636, -0.0032543, 0.0029740
2: 0.0120425, 0.0174979, 0.0116022, 0.0177759, -0.0041509, 0.0045531
3: -0.0015715, 0.0025308, -0.0019025, 0.0027398, -0.0029938, 0.0032798
4: -0.0058291, -0.0020452, -0.0061345, -0.0018524, -0.0038891, 0.0040893
5: 0.0063696, 0.0104644, 0.0060391, 0.0106730, -0.0029764, 0.0032602
6: 0.0075264, 0.0105292, 0.0070805, 0.0106539, -0.0031275, 0.0034487
7: -0.0211164, -0.0122271, -0.0215693, -0.0115098, -0.0056468, 0.0055405
8: 0.9632899, 0.9887587, 0.9619921, 0.9908141, -0.0215879, 0.0196773
9: 0.0009570, 0.0084423, 0.0003529, 0.0088237, -0.0048527, 0.0051614

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A2_A2_A2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108791, upper bound: 0.0116694
time: 0.81 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_A2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108768, upper bound: 0.0116691
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004472, 0.0010101, -0.0004652, 0.0013065, -0.0013470, 0.0011113
1: -0.0007732, 0.0028696, -0.0008575, 0.0033239, -0.0033970, 0.0029359
2: 0.0120425, 0.0174979, 0.0113621, 0.0176242, -0.0040425, 0.0047585
3: -0.0015715, 0.0025308, -0.0020831, 0.0026257, -0.0028909, 0.0034266
4: -0.0058291, -0.0020452, -0.0063011, -0.0019577, -0.0038715, 0.0042558
5: 0.0063696, 0.0104644, 0.0058589, 0.0105591, -0.0028723, 0.0034059
6: 0.0075264, 0.0105292, 0.0068374, 0.0107219, -0.0031955, 0.0036918
7: -0.0211164, -0.0122271, -0.0213221, -0.0111185, -0.0058029, 0.0051220
8: 0.9632899, 0.9887587, 0.9627005, 0.9919351, -0.0225574, 0.0192300
9: 0.0009570, 0.0084423, 0.0000234, 0.0086155, -0.0045446, 0.0053539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A2_A2_A2_B2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108727, upper bound: 0.0116905
time: 0.80 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_A2_B2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108768, upper bound: 0.0116691
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0004665, 0.0009445, -0.0004980, 0.0011645, -0.0012042, 0.0010729
1: -0.0008635, 0.0027691, -0.0010110, 0.0031062, -0.0031441, 0.0027785
2: 0.0121930, 0.0176331, 0.0116881, 0.0178541, -0.0038759, 0.0043949
3: -0.0014583, 0.0026325, -0.0018380, 0.0027986, -0.0027925, 0.0031613
4: -0.0057248, -0.0019515, -0.0060750, -0.0017982, -0.0036477, 0.0039891
5: 0.0064825, 0.0105658, 0.0061036, 0.0107317, -0.0027760, 0.0031418
6: 0.0076788, 0.0104866, 0.0071675, 0.0106296, -0.0029508, 0.0033191
7: -0.0213367, -0.0124724, -0.0216967, -0.0116497, -0.0052622, 0.0051781
8: 0.9626588, 0.9880561, 0.9616272, 0.9904132, -0.0208427, 0.0183794
9: 0.0011635, 0.0086278, 0.0004707, 0.0089309, -0.0045378, 0.0048949

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B2_B1_A1_A1_A1_A1

### Relational analysis result of IS_A1_B2_B2_B1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110990, upper bound: 0.0119602
time: 0.76 seconds

## Relational analysis of IS_A1_B2_B2_B1_A1_A1_A1_A2

### Relational analysis result of IS_A1_B2_B2_B1_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110990, upper bound: 0.0119702
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0004765, 0.0009092, -0.0004980, 0.0011645, -0.0012118, 0.0010364
1: -0.0009105, 0.0027150, -0.0010110, 0.0031062, -0.0031283, 0.0027189
2: 0.0122739, 0.0177036, 0.0116881, 0.0178541, -0.0037897, 0.0043865
3: -0.0013974, 0.0026855, -0.0018380, 0.0027986, -0.0027285, 0.0031615
4: -0.0056686, -0.0019026, -0.0060750, -0.0017982, -0.0035933, 0.0039439
5: 0.0065433, 0.0106187, 0.0061036, 0.0107317, -0.0027124, 0.0031426
6: 0.0077607, 0.0104636, 0.0071675, 0.0106296, -0.0028688, 0.0032962
7: -0.0214515, -0.0126043, -0.0216967, -0.0116497, -0.0053392, 0.0050426
8: 0.9623297, 0.9876783, 0.9616272, 0.9904132, -0.0207890, 0.0179777
9: 0.0012745, 0.0087245, 0.0004707, 0.0089309, -0.0044210, 0.0049265

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B2_B1_A1_A1_A2_B1

### Relational analysis result of IS_A1_B2_B2_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110957, upper bound: 0.0119762
time: 0.79 seconds

## Relational analysis of IS_A1_B2_B2_B1_A1_A1_A2_B2

### Relational analysis result of IS_A1_B2_B2_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110990, upper bound: 0.0119762
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0004615, 0.0009492, -0.0004982, 0.0011695, -0.0012139, 0.0010756
1: -0.0008403, 0.0027763, -0.0010118, 0.0031139, -0.0031657, 0.0027872
2: 0.0121821, 0.0175984, 0.0116766, 0.0178553, -0.0038875, 0.0044305
3: -0.0014665, 0.0026064, -0.0018466, 0.0027995, -0.0027991, 0.0031901
4: -0.0057323, -0.0019755, -0.0060829, -0.0017973, -0.0036575, 0.0040094
5: 0.0064744, 0.0105398, 0.0060949, 0.0107326, -0.0027827, 0.0031707
6: 0.0076678, 0.0104896, 0.0071558, 0.0106328, -0.0029651, 0.0033338
7: -0.0212801, -0.0124547, -0.0216987, -0.0116310, -0.0053388, 0.0051875
8: 0.9628208, 0.9881068, 0.9616215, 0.9904668, -0.0210056, 0.0184355
9: 0.0011486, 0.0085801, 0.0004549, 0.0089326, -0.0045410, 0.0049537

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B2_B1_A1_A2_A1_B1

### Relational analysis result of IS_A1_B2_B2_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111052, upper bound: 0.0119645
time: 0.85 seconds

## Relational analysis of IS_A1_B2_B2_B1_A1_A2_A1_B2

### Relational analysis result of IS_A1_B2_B2_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111127, upper bound: 0.0119645
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0004718, 0.0009138, -0.0004982, 0.0011695, -0.0012210, 0.0010398
1: -0.0008885, 0.0027221, -0.0010118, 0.0031139, -0.0031475, 0.0027275
2: 0.0122634, 0.0176707, 0.0116766, 0.0178553, -0.0037987, 0.0044180
3: -0.0014053, 0.0026607, -0.0018466, 0.0027995, -0.0027344, 0.0031872
4: -0.0056759, -0.0019254, -0.0060829, -0.0017973, -0.0036018, 0.0039646
5: 0.0065354, 0.0105940, 0.0060949, 0.0107326, -0.0027182, 0.0031684
6: 0.0077501, 0.0104666, 0.0071558, 0.0106328, -0.0028828, 0.0033108
7: -0.0213979, -0.0125871, -0.0216987, -0.0116310, -0.0054037, 0.0050480
8: 0.9624835, 0.9877274, 0.9616215, 0.9904668, -0.0209312, 0.0180176
9: 0.0012601, 0.0086793, 0.0004549, 0.0089326, -0.0044253, 0.0049810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B2_B1_A1_A2_A2_B1

### Relational analysis result of IS_A1_B2_B2_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111052, upper bound: 0.0119749
time: 0.71 seconds

## Relational analysis of IS_A1_B2_B2_B1_A1_A2_A2_B2

### Relational analysis result of IS_A1_B2_B2_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111127, upper bound: 0.0119748
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0004411, 0.0010393, -0.0004980, 0.0011645, -0.0012072, 0.0011740
1: -0.0007448, 0.0029144, -0.0010110, 0.0031062, -0.0031919, 0.0030192
2: 0.0119754, 0.0174555, 0.0116881, 0.0178541, -0.0042364, 0.0044454
3: -0.0016219, 0.0024989, -0.0018380, 0.0027986, -0.0030635, 0.0031899
4: -0.0058757, -0.0020747, -0.0060750, -0.0017982, -0.0038978, 0.0040003
5: 0.0063192, 0.0104325, 0.0061036, 0.0107317, -0.0030466, 0.0031694
6: 0.0074584, 0.0105482, 0.0071675, 0.0106296, -0.0031712, 0.0033807
7: -0.0210472, -0.0121178, -0.0216967, -0.0116497, -0.0053260, 0.0057655
8: 0.9634882, 0.9890720, 0.9616272, 0.9904132, -0.0210985, 0.0200625
9: 0.0008649, 0.0083840, 0.0004707, 0.0089309, -0.0050325, 0.0049291

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B2_B1_A2_A1_A1_A1

### Relational analysis result of IS_A1_B2_B2_B1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108641, upper bound: 0.0116480
time: 0.81 seconds

## Relational analysis of IS_A1_B2_B2_B1_A2_A1_A1_A2

### Relational analysis result of IS_A1_B2_B2_B1_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108615, upper bound: 0.0116474
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0004518, 0.0010047, -0.0004980, 0.0011645, -0.0012148, 0.0011400
1: -0.0007948, 0.0028614, -0.0010110, 0.0031062, -0.0031757, 0.0029586
2: 0.0120548, 0.0175302, 0.0116881, 0.0178541, -0.0041487, 0.0044344
3: -0.0015622, 0.0025551, -0.0018380, 0.0027986, -0.0029984, 0.0031869
4: -0.0058206, -0.0020228, -0.0060750, -0.0017982, -0.0038423, 0.0040521
5: 0.0063788, 0.0104886, 0.0061036, 0.0107317, -0.0029818, 0.0031670
6: 0.0075388, 0.0105257, 0.0071675, 0.0106296, -0.0030908, 0.0033582
7: -0.0211691, -0.0122471, -0.0216967, -0.0116497, -0.0054043, 0.0056276
8: 0.9631390, 0.9887015, 0.9616272, 0.9904132, -0.0210366, 0.0196538
9: 0.0009738, 0.0084866, 0.0004707, 0.0089309, -0.0049136, 0.0049652

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B2_B1_A2_A1_A2_A1

### Relational analysis result of IS_A1_B2_B2_B1_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108641, upper bound: 0.0116484
time: 0.71 seconds

## Relational analysis of IS_A1_B2_B2_B1_A2_A1_A2_A2

### Relational analysis result of IS_A1_B2_B2_B1_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108615, upper bound: 0.0116474
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0004362, 0.0010441, -0.0004982, 0.0011695, -0.0012151, 0.0011790
1: -0.0007217, 0.0029218, -0.0010118, 0.0031139, -0.0032127, 0.0030318
2: 0.0119643, 0.0174207, 0.0116766, 0.0178553, -0.0042538, 0.0044751
3: -0.0016303, 0.0024728, -0.0018466, 0.0027995, -0.0030746, 0.0032137
4: -0.0058834, -0.0020988, -0.0060829, -0.0017973, -0.0039116, 0.0039842
5: 0.0063109, 0.0104064, 0.0060949, 0.0107326, -0.0030577, 0.0031935
6: 0.0074472, 0.0105513, 0.0071558, 0.0106328, -0.0031857, 0.0033955
7: -0.0209906, -0.0120997, -0.0216987, -0.0116310, -0.0053791, 0.0057844
8: 0.9636502, 0.9891238, 0.9616215, 0.9904668, -0.0212368, 0.0201458
9: 0.0008497, 0.0083364, 0.0004549, 0.0089326, -0.0050436, 0.0049814

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B2_B1_A2_A2_A1_A1

### Relational analysis result of IS_A1_B2_B2_B1_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108755, upper bound: 0.0116704
time: 0.73 seconds

## Relational analysis of IS_A1_B2_B2_B1_A2_A2_A1_A2

### Relational analysis result of IS_A1_B2_B2_B1_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108737, upper bound: 0.0116689
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0004472, 0.0010101, -0.0004982, 0.0011695, -0.0012224, 0.0011450
1: -0.0007732, 0.0028696, -0.0010118, 0.0031139, -0.0031968, 0.0029698
2: 0.0120425, 0.0174979, 0.0116766, 0.0178553, -0.0041615, 0.0044675
3: -0.0015715, 0.0025308, -0.0018466, 0.0027995, -0.0030073, 0.0032137
4: -0.0058291, -0.0020452, -0.0060829, -0.0017973, -0.0038535, 0.0040377
5: 0.0063696, 0.0104644, 0.0060949, 0.0107326, -0.0029905, 0.0031940
6: 0.0075264, 0.0105292, 0.0071558, 0.0106328, -0.0031065, 0.0033734
7: -0.0211164, -0.0122271, -0.0216987, -0.0116310, -0.0054563, 0.0056393
8: 0.9632899, 0.9887587, 0.9616215, 0.9904668, -0.0211887, 0.0197117
9: 0.0009570, 0.0084423, 0.0004549, 0.0089326, -0.0049232, 0.0050181

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B2_B1_A2_A2_A2_A1

### Relational analysis result of IS_A1_B2_B2_B1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108755, upper bound: 0.0116712
time: 0.76 seconds

## Relational analysis of IS_A1_B2_B2_B1_A2_A2_A2_A2

### Relational analysis result of IS_A1_B2_B2_B1_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108737, upper bound: 0.0116698
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0004617, 0.0009662, -0.0004783, 0.0012478, -0.0012970, 0.0010823
1: -0.0008412, 0.0028024, -0.0009188, 0.0032339, -0.0033576, 0.0028330
2: 0.0121431, 0.0175998, 0.0114969, 0.0177161, -0.0039302, 0.0047113
3: -0.0014958, 0.0026074, -0.0019817, 0.0026948, -0.0028278, 0.0033984
4: -0.0057593, -0.0019746, -0.0062076, -0.0018939, -0.0037682, 0.0042219
5: 0.0064451, 0.0105408, 0.0059600, 0.0106281, -0.0028109, 0.0033783
6: 0.0076283, 0.0105007, 0.0069738, 0.0106837, -0.0030555, 0.0035269
7: -0.0212824, -0.0123911, -0.0214718, -0.0113381, -0.0057474, 0.0051532
8: 0.9628144, 0.9882889, 0.9622716, 0.9913060, -0.0223222, 0.0186524
9: 0.0010950, 0.0085820, 0.0002083, 0.0087416, -0.0045578, 0.0053130

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B2_B2_B1_A1_A1_B1

### Relational analysis result of IS_A1_B2_B2_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109336, upper bound: 0.0117138
time: 0.77 seconds

## Relational analysis of IS_A1_B2_B2_B2_B1_A1_A1_B2

### Relational analysis result of IS_A1_B2_B2_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109347, upper bound: 0.0117025
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0004720, 0.0009310, -0.0004783, 0.0012478, -0.0013051, 0.0010491
1: -0.0008894, 0.0027484, -0.0009188, 0.0032339, -0.0033512, 0.0027832
2: 0.0122239, 0.0176720, 0.0114969, 0.0177161, -0.0038555, 0.0047171
3: -0.0014350, 0.0026617, -0.0019817, 0.0026948, -0.0027662, 0.0034086
4: -0.0057033, -0.0019245, -0.0062076, -0.0018939, -0.0037148, 0.0041892
5: 0.0065058, 0.0105950, 0.0059600, 0.0106281, -0.0027496, 0.0033891
6: 0.0077101, 0.0104778, 0.0069738, 0.0106837, -0.0029736, 0.0035040
7: -0.0214001, -0.0125228, -0.0214718, -0.0113381, -0.0058409, 0.0050399
8: 0.9624772, 0.9879116, 0.9622716, 0.9913060, -0.0223353, 0.0183168
9: 0.0012059, 0.0086811, 0.0002083, 0.0087416, -0.0044506, 0.0053645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B2_B2_B1_A1_A2_B1

### Relational analysis result of IS_A1_B2_B2_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109336, upper bound: 0.0117162
time: 0.85 seconds

## Relational analysis of IS_A1_B2_B2_B2_B1_A1_A2_B2

### Relational analysis result of IS_A1_B2_B2_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109347, upper bound: 0.0117025
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0004364, 0.0010600, -0.0004783, 0.0012478, -0.0012657, 0.0011659
1: -0.0007225, 0.0029462, -0.0009188, 0.0032339, -0.0033217, 0.0029798
2: 0.0119277, 0.0174221, 0.0114969, 0.0177161, -0.0041362, 0.0046239
3: -0.0016578, 0.0024738, -0.0019817, 0.0026948, -0.0029722, 0.0033155
4: -0.0059088, -0.0020978, -0.0062076, -0.0018939, -0.0039288, 0.0041098
5: 0.0062834, 0.0104074, 0.0059600, 0.0106281, -0.0029538, 0.0032942
6: 0.0074101, 0.0105617, 0.0069738, 0.0106837, -0.0032736, 0.0035879
7: -0.0209928, -0.0120401, -0.0214718, -0.0113381, -0.0053622, 0.0054148
8: 0.9636441, 0.9892946, 0.9622716, 0.9913060, -0.0219517, 0.0196325
9: 0.0007995, 0.0083382, 0.0002083, 0.0087416, -0.0047553, 0.0050688

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B2_B2_B1_A2_A1_A1

### Relational analysis result of IS_A1_B2_B2_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108950, upper bound: 0.0116414
time: 0.79 seconds

## Relational analysis of IS_A1_B2_B2_B2_B1_A2_A1_A2

### Relational analysis result of IS_A1_B2_B2_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108886, upper bound: 0.0116390
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0004474, 0.0010254, -0.0004783, 0.0012478, -0.0012728, 0.0011327
1: -0.0007741, 0.0028932, -0.0009188, 0.0032339, -0.0033041, 0.0029191
2: 0.0120071, 0.0174993, 0.0114969, 0.0177161, -0.0040479, 0.0046087
3: -0.0015980, 0.0025318, -0.0019817, 0.0026948, -0.0029083, 0.0033111
4: -0.0058537, -0.0020442, -0.0062076, -0.0018939, -0.0038682, 0.0041634
5: 0.0063430, 0.0104654, 0.0059600, 0.0106281, -0.0028903, 0.0032903
6: 0.0074906, 0.0105392, 0.0069738, 0.0106837, -0.0031932, 0.0035654
7: -0.0211187, -0.0121696, -0.0214718, -0.0113381, -0.0054236, 0.0052798
8: 0.9632834, 0.9889237, 0.9622716, 0.9913060, -0.0218685, 0.0192226
9: 0.0009085, 0.0084442, 0.0002083, 0.0087416, -0.0046530, 0.0050935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B2_B2_B1_A2_A2_A1

### Relational analysis result of IS_A1_B2_B2_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108950, upper bound: 0.0116414
time: 0.69 seconds

## Relational analysis of IS_A1_B2_B2_B2_B1_A2_A2_A2

### Relational analysis result of IS_A1_B2_B2_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108886, upper bound: 0.0116390
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0004618, 0.0009699, -0.0004766, 0.0012603, -0.0013018, 0.0010870
1: -0.0008417, 0.0028080, -0.0009108, 0.0032530, -0.0033555, 0.0028611
2: 0.0121346, 0.0176005, 0.0114682, 0.0177040, -0.0039634, 0.0047021
3: -0.0015022, 0.0026079, -0.0020033, 0.0026858, -0.0028526, 0.0033878
4: -0.0057652, -0.0019741, -0.0062275, -0.0019023, -0.0038076, 0.0042287
5: 0.0064387, 0.0105414, 0.0059385, 0.0106190, -0.0028357, 0.0033675
6: 0.0076197, 0.0105031, 0.0069448, 0.0106918, -0.0030722, 0.0035583
7: -0.0212835, -0.0123773, -0.0214522, -0.0112915, -0.0056986, 0.0051791
8: 0.9628110, 0.9883285, 0.9623277, 0.9914396, -0.0222861, 0.0188128
9: 0.0010834, 0.0085830, 0.0001690, 0.0087251, -0.0045890, 0.0052834

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B2_B2_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_B2_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109127, upper bound: 0.0117275
time: 0.76 seconds

## Relational analysis of IS_A1_B2_B2_B2_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_B2_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109159, upper bound: 0.0117162
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0004721, 0.0009348, -0.0004766, 0.0012603, -0.0013098, 0.0010536
1: -0.0008899, 0.0027542, -0.0009108, 0.0032530, -0.0033482, 0.0028109
2: 0.0122153, 0.0176727, 0.0114682, 0.0177040, -0.0038899, 0.0047070
3: -0.0014415, 0.0026623, -0.0020033, 0.0026858, -0.0027917, 0.0033986
4: -0.0057093, -0.0019240, -0.0062275, -0.0019023, -0.0037559, 0.0041917
5: 0.0064993, 0.0105956, 0.0059385, 0.0106190, -0.0027751, 0.0033788
6: 0.0077014, 0.0104803, 0.0069448, 0.0106918, -0.0029905, 0.0035354
7: -0.0214013, -0.0125087, -0.0214522, -0.0112915, -0.0057984, 0.0050621
8: 0.9624736, 0.9879520, 0.9623277, 0.9914396, -0.0222937, 0.0184822
9: 0.0011940, 0.0086822, 0.0001690, 0.0087251, -0.0044834, 0.0053342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B2_B2_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_B2_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109127, upper bound: 0.0117302
time: 0.79 seconds

## Relational analysis of IS_A1_B2_B2_B2_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_B2_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109159, upper bound: 0.0117162
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0004365, 0.0010638, -0.0004766, 0.0012603, -0.0012712, 0.0011716
1: -0.0007231, 0.0029519, -0.0009108, 0.0032530, -0.0033214, 0.0030109
2: 0.0119192, 0.0174229, 0.0114682, 0.0177040, -0.0041779, 0.0046186
3: -0.0016642, 0.0024744, -0.0020033, 0.0026858, -0.0030011, 0.0033083
4: -0.0059147, -0.0020972, -0.0062275, -0.0019023, -0.0039705, 0.0041302
5: 0.0062770, 0.0104081, 0.0059385, 0.0106190, -0.0029826, 0.0032865
6: 0.0074015, 0.0105641, 0.0069448, 0.0106918, -0.0032904, 0.0036193
7: -0.0209942, -0.0120262, -0.0214522, -0.0112915, -0.0053099, 0.0054533
8: 0.9636400, 0.9893345, 0.9623277, 0.9914396, -0.0219334, 0.0198350
9: 0.0007877, 0.0083394, 0.0001690, 0.0087251, -0.0048014, 0.0050385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B2_B2_B2_A2_A1_A1

### Relational analysis result of IS_A1_B2_B2_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108755, upper bound: 0.0116627
time: 0.75 seconds

## Relational analysis of IS_A1_B2_B2_B2_B2_A2_A1_A2

### Relational analysis result of IS_A1_B2_B2_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108737, upper bound: 0.0116592
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0004475, 0.0010293, -0.0004766, 0.0012603, -0.0012780, 0.0011387
1: -0.0007746, 0.0028991, -0.0009108, 0.0032530, -0.0033029, 0.0029517
2: 0.0119983, 0.0175001, 0.0114682, 0.0177040, -0.0040899, 0.0046035
3: -0.0016047, 0.0025324, -0.0020033, 0.0026858, -0.0029369, 0.0033038
4: -0.0058598, -0.0020437, -0.0062275, -0.0019023, -0.0039101, 0.0041837
5: 0.0063364, 0.0104660, 0.0059385, 0.0106190, -0.0029185, 0.0032827
6: 0.0074816, 0.0105417, 0.0069448, 0.0106918, -0.0032102, 0.0035969
7: -0.0211199, -0.0121551, -0.0214522, -0.0112915, -0.0053791, 0.0053154
8: 0.9632798, 0.9889650, 0.9623277, 0.9914396, -0.0218488, 0.0194212
9: 0.0008963, 0.0084452, 0.0001690, 0.0087251, -0.0046870, 0.0050666

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B2_B2_B2_A2_A2_A1

### Relational analysis result of IS_A1_B2_B2_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108755, upper bound: 0.0116627
time: 0.70 seconds

## Relational analysis of IS_A1_B2_B2_B2_B2_A2_A2_A2

### Relational analysis result of IS_A1_B2_B2_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108737, upper bound: 0.0116592
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0004867, 0.0011974, -0.0004665, 0.0009445, -0.0010630, 0.0012345
1: -0.0009580, 0.0031566, -0.0008635, 0.0027691, -0.0027789, 0.0031888
2: 0.0116126, 0.0177747, 0.0121930, 0.0176331, -0.0044619, 0.0038601
3: -0.0018947, 0.0027389, -0.0014583, 0.0026325, -0.0032117, 0.0027756
4: -0.0061273, -0.0018532, -0.0057248, -0.0019515, -0.0040356, 0.0036869
5: 0.0060469, 0.0106721, 0.0064825, 0.0105658, -0.0031921, 0.0027588
6: 0.0070910, 0.0106510, 0.0076788, 0.0104866, -0.0033955, 0.0029722
7: -0.0215674, -0.0115267, -0.0213367, -0.0124724, -0.0050763, 0.0053713
8: 0.9619977, 0.9907656, 0.9626588, 0.9880561, -0.0183185, 0.0211553
9: 0.0003671, 0.0088220, 0.0011635, 0.0086278, -0.0049868, 0.0044620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120406, upper bound: 0.0111575
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120406, upper bound: 0.0111611
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0004868, 0.0012019, -0.0004615, 0.0009492, -0.0010659, 0.0012436
1: -0.0009588, 0.0031636, -0.0008403, 0.0027763, -0.0027873, 0.0032101
2: 0.0116022, 0.0177759, 0.0121821, 0.0175984, -0.0044970, 0.0038712
3: -0.0019025, 0.0027398, -0.0014665, 0.0026064, -0.0032401, 0.0027835
4: -0.0061345, -0.0018524, -0.0057323, -0.0019755, -0.0040555, 0.0036951
5: 0.0060391, 0.0106730, 0.0064744, 0.0105398, -0.0032207, 0.0027665
6: 0.0070805, 0.0106539, 0.0076678, 0.0104896, -0.0034091, 0.0029861
7: -0.0215693, -0.0115098, -0.0212801, -0.0124547, -0.0050849, 0.0054472
8: 0.9619921, 0.9908141, 0.9628208, 0.9881068, -0.0183717, 0.0213163
9: 0.0003529, 0.0088237, 0.0011486, 0.0085801, -0.0050450, 0.0044690

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120342, upper bound: 0.0111662
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120342, upper bound: 0.0111739
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004899, 0.0011744, -0.0004720, 0.0009310, -0.0010493, 0.0012363
1: -0.0009734, 0.0031214, -0.0008894, 0.0027484, -0.0027403, 0.0031820
2: 0.0116653, 0.0177977, 0.0122239, 0.0176720, -0.0044660, 0.0038126
3: -0.0018551, 0.0027562, -0.0014350, 0.0026617, -0.0032216, 0.0027429
4: -0.0060908, -0.0018373, -0.0057033, -0.0019245, -0.0040053, 0.0036237
5: 0.0060865, 0.0106894, 0.0065058, 0.0105950, -0.0032026, 0.0027265
6: 0.0071444, 0.0106360, 0.0077101, 0.0104778, -0.0033334, 0.0029259
7: -0.0216049, -0.0116125, -0.0214001, -0.0125228, -0.0050436, 0.0055319
8: 0.9618902, 0.9905197, 0.9624772, 0.9879116, -0.0180879, 0.0211594
9: 0.0004394, 0.0088536, 0.0012059, 0.0086811, -0.0050732, 0.0044278

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119749, upper bound: 0.0111360
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119748, upper bound: 0.0111434
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004864, 0.0011819, -0.0004721, 0.0009348, -0.0010544, 0.0012376
1: -0.0009569, 0.0031329, -0.0008899, 0.0027542, -0.0027675, 0.0031794
2: 0.0116482, 0.0177731, 0.0122153, 0.0176727, -0.0044571, 0.0038485
3: -0.0018680, 0.0027377, -0.0014415, 0.0026623, -0.0032126, 0.0027685
4: -0.0061026, -0.0018544, -0.0057093, -0.0019240, -0.0040118, 0.0036607
5: 0.0060736, 0.0106709, 0.0064993, 0.0105956, -0.0031934, 0.0027519
6: 0.0071271, 0.0106409, 0.0077014, 0.0104803, -0.0033532, 0.0029395
7: -0.0215647, -0.0115847, -0.0214013, -0.0125087, -0.0050792, 0.0054661
8: 0.9620054, 0.9905995, 0.9624736, 0.9879520, -0.0182588, 0.0211219
9: 0.0004159, 0.0088198, 0.0011940, 0.0086822, -0.0050388, 0.0044609

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119749, upper bound: 0.0111066
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119748, upper bound: 0.0111127
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004671, 0.0012760, -0.0004617, 0.0009662, -0.0010751, 0.0013286
1: -0.0008664, 0.0032772, -0.0008412, 0.0028024, -0.0028368, 0.0034143
2: 0.0114321, 0.0176375, 0.0121431, 0.0175998, -0.0047961, 0.0039223
3: -0.0020305, 0.0026358, -0.0014958, 0.0026074, -0.0034622, 0.0028103
4: -0.0062525, -0.0019484, -0.0057593, -0.0019746, -0.0042780, 0.0038049
5: 0.0059114, 0.0105691, 0.0064451, 0.0105408, -0.0034420, 0.0027927
6: 0.0069082, 0.0107021, 0.0076283, 0.0105007, -0.0035925, 0.0030738
7: -0.0213438, -0.0112325, -0.0212824, -0.0123911, -0.0050768, 0.0058856
8: 0.9626384, 0.9916085, 0.9628144, 0.9882889, -0.0186335, 0.0227183
9: 0.0001194, 0.0086338, 0.0010950, 0.0085820, -0.0054294, 0.0044952

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118445, upper bound: 0.0110283
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118383, upper bound: 0.0110344
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004648, 0.0012888, -0.0004618, 0.0009699, -0.0010795, 0.0013333
1: -0.0008558, 0.0032969, -0.0008417, 0.0028080, -0.0028649, 0.0034123
2: 0.0114026, 0.0176216, 0.0121346, 0.0176005, -0.0047871, 0.0039536
3: -0.0020526, 0.0026238, -0.0015022, 0.0026079, -0.0034518, 0.0028335
4: -0.0062730, -0.0019595, -0.0057652, -0.0019741, -0.0042877, 0.0038058
5: 0.0058893, 0.0105572, 0.0064387, 0.0105414, -0.0034313, 0.0028157
6: 0.0068784, 0.0107104, 0.0076197, 0.0105031, -0.0036247, 0.0030908
7: -0.0213179, -0.0111845, -0.0212835, -0.0123773, -0.0050935, 0.0058372
8: 0.9627127, 0.9917461, 0.9628110, 0.9883285, -0.0187864, 0.0226831
9: 0.0000790, 0.0086119, 0.0010834, 0.0085830, -0.0054001, 0.0045244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118445, upper bound: 0.0109973
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118383, upper bound: 0.0110035
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004671, 0.0012760, -0.0004720, 0.0009310, -0.0010412, 0.0013507
1: -0.0008664, 0.0032772, -0.0008894, 0.0027484, -0.0027914, 0.0034134
2: 0.0114321, 0.0176375, 0.0122239, 0.0176720, -0.0048126, 0.0038543
3: -0.0020305, 0.0026358, -0.0014350, 0.0026617, -0.0034822, 0.0027592
4: -0.0062525, -0.0019484, -0.0057033, -0.0019245, -0.0042457, 0.0037549
5: 0.0059114, 0.0105691, 0.0065058, 0.0105950, -0.0034628, 0.0027416
6: 0.0069082, 0.0107021, 0.0077101, 0.0104778, -0.0035696, 0.0029920
7: -0.0213438, -0.0112325, -0.0214001, -0.0125228, -0.0049661, 0.0060968
8: 0.9626384, 0.9916085, 0.9624772, 0.9879116, -0.0183161, 0.0227778
9: 0.0001194, 0.0086338, 0.0012059, 0.0086811, -0.0055488, 0.0044020

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117247, upper bound: 0.0109336
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117045, upper bound: 0.0109347
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004648, 0.0012888, -0.0004721, 0.0009348, -0.0010458, 0.0013545
1: -0.0008558, 0.0032969, -0.0008899, 0.0027542, -0.0028217, 0.0034113
2: 0.0114026, 0.0176216, 0.0122153, 0.0176727, -0.0048045, 0.0038890
3: -0.0020526, 0.0026238, -0.0014415, 0.0026623, -0.0034738, 0.0027850
4: -0.0062730, -0.0019595, -0.0057093, -0.0019240, -0.0042527, 0.0037498
5: 0.0058893, 0.0105572, 0.0064993, 0.0105956, -0.0034541, 0.0027672
6: 0.0068784, 0.0107104, 0.0077014, 0.0104803, -0.0036019, 0.0030091
7: -0.0213179, -0.0111845, -0.0214013, -0.0125087, -0.0049882, 0.0060320
8: 0.9627127, 0.9917461, 0.9624736, 0.9879520, -0.0184847, 0.0227435
9: 0.0000790, 0.0086119, 0.0011940, 0.0086822, -0.0055154, 0.0044357

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117348, upper bound: 0.0109127
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117181, upper bound: 0.0109159
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004867, 0.0011974, -0.0004411, 0.0010393, -0.0011641, 0.0012375
1: -0.0009580, 0.0031566, -0.0007448, 0.0029144, -0.0030196, 0.0032366
2: 0.0116126, 0.0177747, 0.0119754, 0.0174555, -0.0045124, 0.0042206
3: -0.0018947, 0.0027389, -0.0016219, 0.0024989, -0.0032403, 0.0030467
4: -0.0061273, -0.0018532, -0.0058757, -0.0020747, -0.0040526, 0.0039369
5: 0.0060469, 0.0106721, 0.0063192, 0.0104325, -0.0032197, 0.0030294
6: 0.0070910, 0.0106510, 0.0074584, 0.0105482, -0.0034572, 0.0031925
7: -0.0215674, -0.0115267, -0.0210472, -0.0121178, -0.0056637, 0.0054351
8: 0.9619977, 0.9907656, 0.9634882, 0.9890720, -0.0200015, 0.0214111
9: 0.0003671, 0.0088220, 0.0008649, 0.0083840, -0.0050210, 0.0049566

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_A1_B2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118019, upper bound: 0.0109590
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A1_B2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118050, upper bound: 0.0109577
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004650, 0.0013005, -0.0004411, 0.0010393, -0.0011362, 0.0013167
1: -0.0008567, 0.0033146, -0.0007448, 0.0029144, -0.0029797, 0.0033774
2: 0.0113760, 0.0176231, 0.0119754, 0.0174555, -0.0047109, 0.0041090
3: -0.0020727, 0.0026249, -0.0016219, 0.0024989, -0.0033823, 0.0029431
4: -0.0062915, -0.0019584, -0.0058757, -0.0020747, -0.0042168, 0.0039173
5: 0.0058693, 0.0105583, 0.0063192, 0.0104325, -0.0033609, 0.0029240
6: 0.0068514, 0.0107180, 0.0074584, 0.0105482, -0.0036968, 0.0032596
7: -0.0213203, -0.0111411, -0.0210472, -0.0121178, -0.0052421, 0.0055217
8: 0.9627057, 0.9918704, 0.9634882, 0.9890720, -0.0195410, 0.0223552
9: 0.0000424, 0.0086140, 0.0008649, 0.0083840, -0.0051974, 0.0046438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_A1_B2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_A1_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118162, upper bound: 0.0109567
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_A1_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118050, upper bound: 0.0109577
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004868, 0.0012019, -0.0004362, 0.0010441, -0.0011693, 0.0012448
1: -0.0009588, 0.0031636, -0.0007217, 0.0029218, -0.0030319, 0.0032571
2: 0.0116022, 0.0177759, 0.0119643, 0.0174207, -0.0045417, 0.0042376
3: -0.0019025, 0.0027398, -0.0016303, 0.0024728, -0.0032637, 0.0030589
4: -0.0061345, -0.0018524, -0.0058834, -0.0020988, -0.0040358, 0.0039492
5: 0.0060391, 0.0106730, 0.0063109, 0.0104064, -0.0032434, 0.0030415
6: 0.0070805, 0.0106539, 0.0074472, 0.0105513, -0.0034708, 0.0032067
7: -0.0215693, -0.0115098, -0.0209906, -0.0120997, -0.0056818, 0.0054875
8: 0.9619921, 0.9908141, 0.9636502, 0.9891238, -0.0200820, 0.0215475
9: 0.0003529, 0.0088237, 0.0008497, 0.0083364, -0.0050727, 0.0049716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.62 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 2.74 + 598.04 = 600.78 seconds
