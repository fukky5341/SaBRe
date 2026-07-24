## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0012732


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0015982, 0.0015982)
1: (0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0002309, 0.0002309)
2: (0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0008836, 0.0008836)
3: (-0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0009138, 0.0009138)
4: (-0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0009893, 0.0009893)
5: (0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0009362, 0.0009362)
6: (-0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0037145, 0.0037145)
7: (-0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0050589, 0.0050589)
8: (0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0035636, 0.0035636)
9: (-0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0032348, 0.0032348)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.36 + 1.60 = 2.97 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0020450, upper bound: 0.0020450

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019372, upper bound: 0.0019840
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019840, upper bound: 0.0019372
time: 0.78 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.68 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.68
Output dim: 8, lower bound: -0.0019372, upper bound: 0.0019840
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.68
Output dim: 8, lower bound: -0.0019840, upper bound: 0.0019372

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0014384, 0.0014560
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0002078, 0.0002104
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0008050, 0.0007952
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0008326, 0.0008225
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008904, 0.0009013
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0008529, 0.0008426
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0033842, 0.0033431
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0045530, 0.0046090
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0032073, 0.0032467
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0029471, 0.0029113

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018701, upper bound: 0.0019134
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018694, upper bound: 0.0019134
time: 0.78 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0014560, 0.0014384
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0002104, 0.0002078
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007952, 0.0008050
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0008225, 0.0008326
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0009013, 0.0008904
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0008426, 0.0008529
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0033431, 0.0033842
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0046090, 0.0045530
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0032467, 0.0032073
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0029113, 0.0029471

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019329, upper bound: 0.0018829
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019329, upper bound: 0.0018899
time: 0.76 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.74 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 8, lower bound: -0.0018701, upper bound: 0.0019134
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 8, lower bound: -0.0018694, upper bound: 0.0019134
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 8, lower bound: -0.0019329, upper bound: 0.0018829
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 8, lower bound: -0.0019329, upper bound: 0.0018899

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0013975, 0.0014182
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0002019, 0.0002049
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007841, 0.0007727
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0008109, 0.0007991
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008651, 0.0008779
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0008308, 0.0008187
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0032962, 0.0032482
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0044238, 0.0044892
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0031162, 0.0031623
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0028705, 0.0028287

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017494, upper bound: 0.0017990
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017568, upper bound: 0.0017925
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0014005, 0.0014148
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0002023, 0.0002044
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007822, 0.0007743
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0008090, 0.0008008
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008669, 0.0008758
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0008288, 0.0008204
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0032883, 0.0032551
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0044332, 0.0044784
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0031229, 0.0031547
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0028636, 0.0028347

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017778, upper bound: 0.0018301
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017877, upper bound: 0.0018216
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0014366, 0.0014206
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0002075, 0.0002052
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007854, 0.0007942
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0008123, 0.0008214
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008893, 0.0008794
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0008322, 0.0008415
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0033019, 0.0033390
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0045474, 0.0044969
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0032033, 0.0031677
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0028755, 0.0029077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018612, upper bound: 0.0018134
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018612, upper bound: 0.0018138
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0014383, 0.0014204
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0002078, 0.0002052
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007853, 0.0007952
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0008122, 0.0008224
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008903, 0.0008792
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0008320, 0.0008426
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0033013, 0.0033430
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0045529, 0.0044961
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0032071, 0.0031672
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0028749, 0.0029112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018846, upper bound: 0.0018271
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018751, upper bound: 0.0018439
time: 0.72 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.91 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 8, lower bound: -0.0017494, upper bound: 0.0017990
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 8, lower bound: -0.0017568, upper bound: 0.0017925
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 8, lower bound: -0.0017778, upper bound: 0.0018301
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 8, lower bound: -0.0017877, upper bound: 0.0018216
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 8, lower bound: -0.0018612, upper bound: 0.0018134
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 8, lower bound: -0.0018612, upper bound: 0.0018138
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 8, lower bound: -0.0018846, upper bound: 0.0018271
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 8, lower bound: -0.0018751, upper bound: 0.0018439

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012852, 0.0013044
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001857, 0.0001885
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007212, 0.0007105
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007459, 0.0007349
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007955, 0.0008075
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007641, 0.0007529
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0030319, 0.0029871
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0040682, 0.0041291
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028657, 0.0029087
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0026403, 0.0026013

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016988, upper bound: 0.0017469
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016935, upper bound: 0.0017469
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012837, 0.0013058
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001855, 0.0001887
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007220, 0.0007097
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007467, 0.0007340
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007946, 0.0008083
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007650, 0.0007520
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0030351, 0.0029837
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0040635, 0.0041335
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028624, 0.0029118
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0026431, 0.0025983

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017097, upper bound: 0.0017357
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016985, upper bound: 0.0017446
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0011739, 0.0011894
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001696, 0.0001718
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0006576, 0.0006490
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006801, 0.0006712
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007266, 0.0007362
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006967, 0.0006876
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0027645, 0.0027284
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0037158, 0.0037649
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0026175, 0.0026521
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0024074, 0.0023760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016578, upper bound: 0.0017154
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016657, upper bound: 0.0017092
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0011752, 0.0011881
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001698, 0.0001716
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0006569, 0.0006497
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006794, 0.0006720
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007275, 0.0007355
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006960, 0.0006884
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0027615, 0.0027315
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0037200, 0.0037610
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0026205, 0.0026493
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0024049, 0.0023787

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012153, upper bound: 0.0012235
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012153, upper bound: 0.0012235
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0013946, 0.0013826
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0002015, 0.0001997
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007644, 0.0007710
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007906, 0.0007974
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008633, 0.0008558
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0008099, 0.0008169
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0032135, 0.0032414
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0044145, 0.0043765
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0031096, 0.0030829
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0027985, 0.0028227

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018172, upper bound: 0.0017552
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018057, upper bound: 0.0017705
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0013985, 0.0013796
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0002020, 0.0001993
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007628, 0.0007732
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007889, 0.0007997
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008657, 0.0008540
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0008082, 0.0008193
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0032066, 0.0032506
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0044270, 0.0043671
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0031184, 0.0030763
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0027924, 0.0028307

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018476, upper bound: 0.0018059
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018533, upper bound: 0.0018050
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0014282, 0.0014096
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0002063, 0.0002036
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007793, 0.0007896
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0008060, 0.0008167
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008841, 0.0008726
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0008257, 0.0008366
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0032763, 0.0033196
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0045210, 0.0044621
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0031847, 0.0031432
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0028532, 0.0028908

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018716, upper bound: 0.0018197
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018775, upper bound: 0.0018198
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0014275, 0.0014112
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0002062, 0.0002039
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007802, 0.0007892
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0008069, 0.0008163
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008837, 0.0008736
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0008267, 0.0008362
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0032801, 0.0033180
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0045188, 0.0044672
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0031832, 0.0031468
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0028564, 0.0028895

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018414, upper bound: 0.0018100
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018419, upper bound: 0.0018100
time: 0.82 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.99 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 8, lower bound: -0.0016988, upper bound: 0.0017469
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 8, lower bound: -0.0016935, upper bound: 0.0017469
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 8, lower bound: -0.0017097, upper bound: 0.0017357
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 8, lower bound: -0.0016985, upper bound: 0.0017446
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 8, lower bound: -0.0016578, upper bound: 0.0017154
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 8, lower bound: -0.0016657, upper bound: 0.0017092
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.99
Output dim: 8, lower bound: -0.0012153, upper bound: 0.0012235
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.99
Output dim: 8, lower bound: -0.0012153, upper bound: 0.0012235
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 8, lower bound: -0.0018172, upper bound: 0.0017552
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 8, lower bound: -0.0018057, upper bound: 0.0017705
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 8, lower bound: -0.0018476, upper bound: 0.0018059
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 8, lower bound: -0.0018533, upper bound: 0.0018050
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 8, lower bound: -0.0018716, upper bound: 0.0018197
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 8, lower bound: -0.0018775, upper bound: 0.0018198
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 8, lower bound: -0.0018414, upper bound: 0.0018100
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 8, lower bound: -0.0018419, upper bound: 0.0018100

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012670, 0.0012865
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001830, 0.0001859
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007113, 0.0007005
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007356, 0.0007245
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007843, 0.0007964
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007536, 0.0007422
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029902, 0.0029449
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0040107, 0.0040725
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028252, 0.0028687
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0026040, 0.0025646

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016580, upper bound: 0.0016939
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016438, upper bound: 0.0017042
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012673, 0.0012842
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001831, 0.0001855
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007100, 0.0007006
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007343, 0.0007246
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007845, 0.0007949
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007523, 0.0007424
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029847, 0.0029455
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0040115, 0.0040649
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028258, 0.0028634
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025992, 0.0025650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016510, upper bound: 0.0017027
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016510, upper bound: 0.0017027
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012740, 0.0012952
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001841, 0.0001871
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007161, 0.0007044
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007406, 0.0007285
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007886, 0.0008017
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007587, 0.0007463
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0030104, 0.0029612
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0040329, 0.0040999
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028408, 0.0028880
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0026216, 0.0025787

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017018, upper bound: 0.0017282
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017023, upper bound: 0.0017188
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012731, 0.0012959
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001839, 0.0001872
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007165, 0.0007039
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007410, 0.0007280
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007881, 0.0008022
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007591, 0.0007458
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0030120, 0.0029590
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0040299, 0.0041020
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028387, 0.0028896
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0026229, 0.0025768

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016908, upper bound: 0.0017372
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016908, upper bound: 0.0017323
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0010603, 0.0010751
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001532, 0.0001553
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0005944, 0.0005862
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006147, 0.0006063
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0006564, 0.0006655
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006298, 0.0006211
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0024987, 0.0024645
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0033564, 0.0034031
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0023643, 0.0023972
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0021760, 0.0021462

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0010927, upper bound: 0.0011169
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0010927, upper bound: 0.0011169
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0010587, 0.0010758
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001530, 0.0001554
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0005948, 0.0005853
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006152, 0.0006054
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0006554, 0.0006660
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006302, 0.0006202
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0025005, 0.0024608
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0033514, 0.0034055
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0023608, 0.0023989
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0021776, 0.0021430

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016571, upper bound: 0.0017012
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016575, upper bound: 0.0016949
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0013848, 0.0013718
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0002001, 0.0001982
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007584, 0.0007656
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007844, 0.0007918
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008572, 0.0008492
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0008036, 0.0008112
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0031885, 0.0032186
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0043835, 0.0043424
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0030878, 0.0030589
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0027767, 0.0028029

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018058, upper bound: 0.0017480
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018101, upper bound: 0.0017477
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0013838, 0.0013732
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001999, 0.0001984
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007592, 0.0007651
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007852, 0.0007913
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008566, 0.0008500
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0008044, 0.0008106
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0031917, 0.0032163
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0043804, 0.0043468
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0030856, 0.0030620
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0027795, 0.0028009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017103, upper bound: 0.0016880
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017268, upper bound: 0.0016800
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0014148, 0.0013994
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0002044, 0.0002022
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007737, 0.0007822
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0008002, 0.0008090
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008758, 0.0008662
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0008198, 0.0008288
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0032525, 0.0032885
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0044786, 0.0044297
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0031548, 0.0031204
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0028325, 0.0028638

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017356, upper bound: 0.0017286
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017740, upper bound: 0.0017144
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0014180, 0.0013959
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0002049, 0.0002017
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007718, 0.0007840
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007982, 0.0008108
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008777, 0.0008641
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0008177, 0.0008306
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0032445, 0.0032957
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0044885, 0.0044188
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0031618, 0.0031127
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0028255, 0.0028701

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017324, upper bound: 0.0016926
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017388, upper bound: 0.0016847
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0014442, 0.0014293
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0002086, 0.0002065
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007902, 0.0007985
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0008173, 0.0008258
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008940, 0.0008847
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0008373, 0.0008460
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0033220, 0.0033568
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0045716, 0.0045243
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0032203, 0.0031870
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0028930, 0.0029232

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018049, upper bound: 0.0017547
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018048, upper bound: 0.0017552
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0014478, 0.0014256
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0002092, 0.0002060
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007882, 0.0008004
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0008152, 0.0008278
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008962, 0.0008825
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0008351, 0.0008481
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0033135, 0.0033650
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0045828, 0.0045127
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0032283, 0.0031788
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0028855, 0.0029304

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017870, upper bound: 0.0017460
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017935, upper bound: 0.0017231
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0014236, 0.0014073
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0002057, 0.0002033
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007781, 0.0007871
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0008047, 0.0008140
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008812, 0.0008711
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0008244, 0.0008339
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0032709, 0.0033088
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0045063, 0.0044547
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0031743, 0.0031380
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0028485, 0.0028814

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017737, upper bound: 0.0017429
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017739, upper bound: 0.0017432
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0014243, 0.0014073
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0002058, 0.0002033
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007780, 0.0007874
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0008047, 0.0008144
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008816, 0.0008711
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0008244, 0.0008343
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0032709, 0.0033104
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0045084, 0.0044546
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0031758, 0.0031379
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0028484, 0.0028828

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017086, upper bound: 0.0016811
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017154, upper bound: 0.0016805
time: 0.82 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.97 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 8, lower bound: -0.0016580, upper bound: 0.0016939
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 8, lower bound: -0.0016438, upper bound: 0.0017042
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 8, lower bound: -0.0016510, upper bound: 0.0017027
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 8, lower bound: -0.0016510, upper bound: 0.0017027
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 8, lower bound: -0.0017018, upper bound: 0.0017282
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 8, lower bound: -0.0017023, upper bound: 0.0017188
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 8, lower bound: -0.0016908, upper bound: 0.0017372
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 8, lower bound: -0.0016908, upper bound: 0.0017323
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.97
Output dim: 8, lower bound: -0.0010927, upper bound: 0.0011169
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.97
Output dim: 8, lower bound: -0.0010927, upper bound: 0.0011169
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 8, lower bound: -0.0016571, upper bound: 0.0017012
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 8, lower bound: -0.0016575, upper bound: 0.0016949
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 8, lower bound: -0.0018058, upper bound: 0.0017480
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 8, lower bound: -0.0018101, upper bound: 0.0017477
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 8, lower bound: -0.0017103, upper bound: 0.0016880
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 8, lower bound: -0.0017268, upper bound: 0.0016800
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 8, lower bound: -0.0017356, upper bound: 0.0017286
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 8, lower bound: -0.0017740, upper bound: 0.0017144
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 8, lower bound: -0.0017324, upper bound: 0.0016926
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 8, lower bound: -0.0017388, upper bound: 0.0016847
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 8, lower bound: -0.0018049, upper bound: 0.0017547
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 8, lower bound: -0.0018048, upper bound: 0.0017552
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 8, lower bound: -0.0017870, upper bound: 0.0017460
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 8, lower bound: -0.0017935, upper bound: 0.0017231
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 8, lower bound: -0.0017737, upper bound: 0.0017429
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 8, lower bound: -0.0017739, upper bound: 0.0017432
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 8, lower bound: -0.0017086, upper bound: 0.0016811
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 8, lower bound: -0.0017154, upper bound: 0.0016805

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012582, 0.0012758
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001818, 0.0001843
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007053, 0.0006956
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007295, 0.0007195
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007789, 0.0007897
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007473, 0.0007371
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029652, 0.0029245
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0039829, 0.0040383
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028056, 0.0028447
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025822, 0.0025467

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015609, upper bound: 0.0016163
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015761, upper bound: 0.0015851
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012563, 0.0012767
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001815, 0.0001844
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007059, 0.0006945
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007300, 0.0007183
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007776, 0.0007903
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007479, 0.0007359
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029674, 0.0029199
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0039766, 0.0040413
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028012, 0.0028468
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025841, 0.0025428

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016360, upper bound: 0.0016967
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016363, upper bound: 0.0016942
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012629, 0.0012799
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001825, 0.0001849
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007076, 0.0006982
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007318, 0.0007221
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007818, 0.0007923
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007497, 0.0007398
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029748, 0.0029354
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0039977, 0.0040514
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028161, 0.0028539
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025906, 0.0025563

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016411, upper bound: 0.0016940
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016425, upper bound: 0.0016905
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012631, 0.0012798
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001825, 0.0001849
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007076, 0.0006983
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007318, 0.0007222
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007819, 0.0007922
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007497, 0.0007399
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029746, 0.0029357
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0039982, 0.0040512
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028164, 0.0028537
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025904, 0.0025565

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016140, upper bound: 0.0016563
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016017, upper bound: 0.0016653
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012854, 0.0013100
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001857, 0.0001893
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007243, 0.0007107
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007491, 0.0007350
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007957, 0.0008109
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007674, 0.0007530
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0030447, 0.0029877
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0040690, 0.0041467
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028663, 0.0029210
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0026515, 0.0026018

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016576, upper bound: 0.0016887
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016576, upper bound: 0.0016887
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012875, 0.0013066
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001860, 0.0001888
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007224, 0.0007118
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007471, 0.0007362
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007970, 0.0008088
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007654, 0.0007542
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0030369, 0.0029925
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0040756, 0.0041360
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028709, 0.0029135
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0026447, 0.0026060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016580, upper bound: 0.0016793
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016580, upper bound: 0.0016790
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012845, 0.0013108
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001856, 0.0001894
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007247, 0.0007102
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007495, 0.0007345
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007951, 0.0008114
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007679, 0.0007525
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0030467, 0.0029855
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0040660, 0.0041494
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028642, 0.0029229
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0026532, 0.0025999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016079, upper bound: 0.0016695
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016212, upper bound: 0.0016310
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012863, 0.0013073
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001858, 0.0001889
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007228, 0.0007111
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007475, 0.0007355
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007962, 0.0008092
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007658, 0.0007535
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0030385, 0.0029896
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0040716, 0.0041381
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028681, 0.0029150
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0026460, 0.0026035

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015947, upper bound: 0.0016486
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016154, upper bound: 0.0016415
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0010829, 0.0011036
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001565, 0.0001594
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0006101, 0.0005987
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006310, 0.0006192
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0006704, 0.0006831
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006465, 0.0006344
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0025650, 0.0025170
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0034280, 0.0034933
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0024148, 0.0024607
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0022337, 0.0021920

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016061, upper bound: 0.0016498
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016011, upper bound: 0.0016498
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0010857, 0.0011000
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001569, 0.0001589
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0006082, 0.0006003
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006290, 0.0006208
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0006721, 0.0006809
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006444, 0.0006360
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0025568, 0.0025235
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0034368, 0.0034821
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0024210, 0.0024529
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0022266, 0.0021976

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016063, upper bound: 0.0016439
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016020, upper bound: 0.0016442
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0014009, 0.0013916
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0002024, 0.0002010
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007694, 0.0007745
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007957, 0.0008011
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008672, 0.0008614
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0008152, 0.0008207
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0032345, 0.0032561
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0044346, 0.0044051
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0031238, 0.0031030
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0028167, 0.0028356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017717, upper bound: 0.0017162
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017717, upper bound: 0.0017162
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0014041, 0.0013879
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0002029, 0.0002005
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007674, 0.0007763
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007936, 0.0008029
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008692, 0.0008592
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0008131, 0.0008225
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0032260, 0.0032635
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0044447, 0.0043935
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0031309, 0.0030949
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0028093, 0.0028420

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016907, upper bound: 0.0016409
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016968, upper bound: 0.0016292
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0011536, 0.0011435
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001667, 0.0001652
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0006322, 0.0006378
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006539, 0.0006597
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007141, 0.0007079
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006699, 0.0006758
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0026579, 0.0026814
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0036518, 0.0036199
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0025724, 0.0025499
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0023146, 0.0023351

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016786, upper bound: 0.0016556
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016786, upper bound: 0.0016556
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0011552, 0.0011430
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001669, 0.0001651
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0006320, 0.0006387
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006536, 0.0006606
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007151, 0.0007076
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006696, 0.0006767
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0026568, 0.0026850
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0036567, 0.0036183
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0025759, 0.0025488
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0023136, 0.0023382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016082, upper bound: 0.0015693
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016136, upper bound: 0.0015609
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0013942, 0.0013825
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0002014, 0.0001997
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007644, 0.0007708
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007905, 0.0007972
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008630, 0.0008558
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0008099, 0.0008167
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0032134, 0.0032405
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0044133, 0.0043764
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0031088, 0.0030828
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0027984, 0.0028220

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016143, upper bound: 0.0016143
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016258, upper bound: 0.0016070
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0013999, 0.0013787
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0002022, 0.0001992
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007623, 0.0007740
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007884, 0.0008005
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008666, 0.0008535
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0008077, 0.0008201
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0032046, 0.0032538
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0044313, 0.0043644
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0031215, 0.0030743
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0027907, 0.0028335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016508, upper bound: 0.0016023
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016584, upper bound: 0.0015939
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0013004, 0.0012770
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001879, 0.0001845
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007060, 0.0007190
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007302, 0.0007436
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008050, 0.0007905
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007480, 0.0007618
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029680, 0.0030226
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0041165, 0.0040422
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028998, 0.0028474
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025847, 0.0026322

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016891, upper bound: 0.0016477
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016891, upper bound: 0.0016477
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012976, 0.0012784
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001875, 0.0001847
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007068, 0.0007174
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007310, 0.0007420
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008033, 0.0007914
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007489, 0.0007601
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029714, 0.0030160
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0041076, 0.0040468
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028935, 0.0028506
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025876, 0.0026265

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016940, upper bound: 0.0016411
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016940, upper bound: 0.0016411
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0014028, 0.0013914
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0002027, 0.0002010
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007692, 0.0007756
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007956, 0.0008021
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008683, 0.0008613
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0008151, 0.0008218
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0032339, 0.0032605
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0044405, 0.0044043
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0031280, 0.0031025
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0028162, 0.0028394

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017141, upper bound: 0.0016802
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017213, upper bound: 0.0016569
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0014063, 0.0013881
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0002032, 0.0002005
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007674, 0.0007775
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007937, 0.0008041
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008705, 0.0008593
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0008131, 0.0008238
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0032263, 0.0032686
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0044516, 0.0043940
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0031358, 0.0030952
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0028096, 0.0028465

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017140, upper bound: 0.0016806
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017213, upper bound: 0.0016570
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012287, 0.0012071
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001775, 0.0001744
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0006674, 0.0006793
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006903, 0.0007026
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007606, 0.0007472
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007071, 0.0007198
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0028057, 0.0028558
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0038894, 0.0038212
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0027398, 0.0026917
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0024433, 0.0024870

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017186, upper bound: 0.0016802
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017187, upper bound: 0.0016805
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012304, 0.0012065
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001778, 0.0001743
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0006671, 0.0006802
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006899, 0.0007035
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007616, 0.0007469
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007068, 0.0007207
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0028043, 0.0028597
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0038947, 0.0038192
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0027435, 0.0026903
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0024421, 0.0024904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017265, upper bound: 0.0016568
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017264, upper bound: 0.0016570
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0013821, 0.0013692
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001997, 0.0001978
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007570, 0.0007641
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007829, 0.0007903
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008556, 0.0008476
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0008021, 0.0008096
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0031825, 0.0032124
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0043750, 0.0043343
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0030819, 0.0030532
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0027715, 0.0027975

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017581, upper bound: 0.0017351
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017660, upper bound: 0.0017339
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0013855, 0.0013666
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0002002, 0.0001974
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007555, 0.0007660
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007814, 0.0007923
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008577, 0.0008459
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0008005, 0.0008116
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0031763, 0.0032203
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0043858, 0.0043258
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0030895, 0.0030472
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0027661, 0.0028044

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016508, upper bound: 0.0016225
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016563, upper bound: 0.0016175
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0014041, 0.0013905
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0002028, 0.0002009
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007688, 0.0007763
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007951, 0.0008028
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008691, 0.0008607
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0008145, 0.0008225
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0032319, 0.0032634
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0044445, 0.0044015
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0031308, 0.0031005
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0028145, 0.0028419

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016177, upper bound: 0.0015997
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016292, upper bound: 0.0015893
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0014097, 0.0013871
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0002037, 0.0002004
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007669, 0.0007794
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007931, 0.0008061
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008726, 0.0008586
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0008125, 0.0008258
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0032239, 0.0032765
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0044624, 0.0043907
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0031434, 0.0030929
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0028075, 0.0028534

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016296, upper bound: 0.0016016
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016296, upper bound: 0.0016021
time: 0.89 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.01 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0015609, upper bound: 0.0016163
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0015761, upper bound: 0.0015851
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016360, upper bound: 0.0016967
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016363, upper bound: 0.0016942
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016411, upper bound: 0.0016940
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016425, upper bound: 0.0016905
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016140, upper bound: 0.0016563
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016017, upper bound: 0.0016653
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016576, upper bound: 0.0016887
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016576, upper bound: 0.0016887
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016580, upper bound: 0.0016793
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016580, upper bound: 0.0016790
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016079, upper bound: 0.0016695
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016212, upper bound: 0.0016310
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0015947, upper bound: 0.0016486
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016154, upper bound: 0.0016415
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016061, upper bound: 0.0016498
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016011, upper bound: 0.0016498
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016063, upper bound: 0.0016439
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016020, upper bound: 0.0016442
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0017717, upper bound: 0.0017162
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0017717, upper bound: 0.0017162
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016907, upper bound: 0.0016409
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016968, upper bound: 0.0016292
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016786, upper bound: 0.0016556
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016786, upper bound: 0.0016556
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016082, upper bound: 0.0015693
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016136, upper bound: 0.0015609
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016143, upper bound: 0.0016143
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016258, upper bound: 0.0016070
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016508, upper bound: 0.0016023
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016584, upper bound: 0.0015939
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016891, upper bound: 0.0016477
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016891, upper bound: 0.0016477
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016940, upper bound: 0.0016411
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016940, upper bound: 0.0016411
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0017141, upper bound: 0.0016802
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0017213, upper bound: 0.0016569
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0017140, upper bound: 0.0016806
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0017213, upper bound: 0.0016570
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0017186, upper bound: 0.0016802
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0017187, upper bound: 0.0016805
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0017265, upper bound: 0.0016568
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0017264, upper bound: 0.0016570
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0017581, upper bound: 0.0017351
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0017660, upper bound: 0.0017339
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016508, upper bound: 0.0016225
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016563, upper bound: 0.0016175
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016177, upper bound: 0.0015997
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016292, upper bound: 0.0015893
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016296, upper bound: 0.0016016
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 8, lower bound: -0.0016296, upper bound: 0.0016021

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012383, 0.0012615
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001789, 0.0001823
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0006975, 0.0006846
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007213, 0.0007081
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007665, 0.0007809
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007390, 0.0007254
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029321, 0.0028781
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0039197, 0.0039932
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0027611, 0.0028129
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025534, 0.0025064

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015521, upper bound: 0.0016080
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015530, upper bound: 0.0016049
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012419, 0.0012558
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001794, 0.0001814
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0006943, 0.0006866
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007181, 0.0007101
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007687, 0.0007774
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007357, 0.0007275
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029189, 0.0028864
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0039310, 0.0039752
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0027691, 0.0028002
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025419, 0.0025136

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015677, upper bound: 0.0015772
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015680, upper bound: 0.0015759
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012672, 0.0012900
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001831, 0.0001864
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007132, 0.0007006
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007376, 0.0007246
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007844, 0.0007985
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007557, 0.0007423
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029983, 0.0029454
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0040113, 0.0040834
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028257, 0.0028764
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0026110, 0.0025650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015380, upper bound: 0.0016129
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015617, upper bound: 0.0016059
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012706, 0.0012877
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001836, 0.0001860
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007119, 0.0007025
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007363, 0.0007265
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007865, 0.0007971
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007543, 0.0007443
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029929, 0.0029532
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0040220, 0.0040760
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028332, 0.0028712
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0026063, 0.0025718

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015963, upper bound: 0.0016541
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015963, upper bound: 0.0016541
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012741, 0.0012932
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001841, 0.0001868
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007150, 0.0007044
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007395, 0.0007285
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007887, 0.0008005
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007576, 0.0007464
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0030058, 0.0029614
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0040331, 0.0040936
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028410, 0.0028836
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0026176, 0.0025789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016042, upper bound: 0.0016479
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015931, upper bound: 0.0016571
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012776, 0.0012911
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001846, 0.0001865
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007138, 0.0007063
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007382, 0.0007305
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007908, 0.0007992
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007563, 0.0007484
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0030008, 0.0029694
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0040441, 0.0040868
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028487, 0.0028788
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0026132, 0.0025859

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015020, upper bound: 0.0015281
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015022, upper bound: 0.0015269
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012539, 0.0012690
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001811, 0.0001833
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007016, 0.0006932
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007256, 0.0007170
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007762, 0.0007856
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007434, 0.0007345
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029496, 0.0029144
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0039691, 0.0040171
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0027959, 0.0028297
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025686, 0.0025379

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014666, upper bound: 0.0014914
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014666, upper bound: 0.0014901
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012523, 0.0012700
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001809, 0.0001835
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007022, 0.0006924
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007262, 0.0007161
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007752, 0.0007862
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007440, 0.0007336
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029519, 0.0029107
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0039641, 0.0040202
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0027924, 0.0028319
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025706, 0.0025347

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015931, upper bound: 0.0016571
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015934, upper bound: 0.0016541
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012811, 0.0013063
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001851, 0.0001887
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007222, 0.0007083
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007470, 0.0007326
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007930, 0.0008086
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007652, 0.0007505
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0030363, 0.0029777
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0040553, 0.0041351
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028567, 0.0029129
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0026441, 0.0025931

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015661, upper bound: 0.0016096
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015785, upper bound: 0.0015935
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012812, 0.0013057
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001851, 0.0001886
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007219, 0.0007084
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007466, 0.0007326
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007931, 0.0008082
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007649, 0.0007505
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0030347, 0.0029779
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0040557, 0.0041330
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028569, 0.0029114
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0026428, 0.0025933

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015929, upper bound: 0.0016426
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016108, upper bound: 0.0016425
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012832, 0.0013030
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001854, 0.0001882
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007204, 0.0007095
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007451, 0.0007337
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007943, 0.0008066
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007633, 0.0007517
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0030285, 0.0029825
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0040619, 0.0041245
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028613, 0.0029054
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0026373, 0.0025973

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015663, upper bound: 0.0016023
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015785, upper bound: 0.0015859
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012834, 0.0013023
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001854, 0.0001881
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007200, 0.0007095
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007447, 0.0007338
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007944, 0.0008061
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007629, 0.0007518
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0030269, 0.0029829
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0040624, 0.0041223
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028617, 0.0029039
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0026359, 0.0025976

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015309, upper bound: 0.0015519
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015315, upper bound: 0.0015449
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012642, 0.0012961
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001826, 0.0001872
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007166, 0.0006990
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007411, 0.0007229
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007826, 0.0008023
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007592, 0.0007406
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0030124, 0.0029384
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0040019, 0.0041027
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028190, 0.0028900
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0026233, 0.0025589

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015244, upper bound: 0.0015630
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015244, upper bound: 0.0015630
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012676, 0.0012906
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001831, 0.0001864
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007135, 0.0007008
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007380, 0.0007248
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007847, 0.0007989
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007560, 0.0007426
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029996, 0.0029463
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0040127, 0.0040852
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028266, 0.0028777
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0026122, 0.0025658

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015248, upper bound: 0.0015562
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015248, upper bound: 0.0015562
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0010674, 0.0010900
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001542, 0.0001575
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0006026, 0.0005901
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006233, 0.0006104
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0006608, 0.0006747
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006385, 0.0006253
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0025335, 0.0024810
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0033789, 0.0034504
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0023802, 0.0024305
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0022063, 0.0021605

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015508, upper bound: 0.0016022
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015459, upper bound: 0.0016028
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0010733, 0.0010884
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001551, 0.0001572
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0006018, 0.0005934
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006224, 0.0006137
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0006644, 0.0006738
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006376, 0.0006288
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0025299, 0.0024947
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0033976, 0.0034454
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0023933, 0.0024270
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0022031, 0.0021725

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015318, upper bound: 0.0015732
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015457, upper bound: 0.0015362
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0010633, 0.0010847
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001536, 0.0001567
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0005997, 0.0005878
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006202, 0.0006080
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0006582, 0.0006715
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006354, 0.0006229
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0025212, 0.0024713
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0033657, 0.0034336
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0023709, 0.0024187
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0021955, 0.0021521

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015672, upper bound: 0.0016009
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015507, upper bound: 0.0016069
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0010641, 0.0010826
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001537, 0.0001564
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0005986, 0.0005883
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006191, 0.0006085
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0006587, 0.0006702
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006342, 0.0006233
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0025163, 0.0024732
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0033683, 0.0034270
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0023727, 0.0024140
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0021913, 0.0021538

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015089, upper bound: 0.0015740
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015219, upper bound: 0.0015376
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0010661, 0.0010812
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001540, 0.0001562
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0005978, 0.0005894
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006182, 0.0006096
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0006599, 0.0006693
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006334, 0.0006245
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0025130, 0.0024778
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0033746, 0.0034225
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0023771, 0.0024109
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0021884, 0.0021578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015594, upper bound: 0.0016015
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015594, upper bound: 0.0016015
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0010669, 0.0010792
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001541, 0.0001559
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0005967, 0.0005898
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006171, 0.0006100
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0006604, 0.0006681
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006322, 0.0006250
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0025084, 0.0024797
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0033771, 0.0034162
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0023789, 0.0024064
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0021844, 0.0021594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015101, upper bound: 0.0015691
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015224, upper bound: 0.0015335
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0013971, 0.0013881
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0002018, 0.0002005
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007674, 0.0007724
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007937, 0.0007989
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008648, 0.0008592
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0008131, 0.0008184
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0032262, 0.0032472
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0044224, 0.0043938
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0031152, 0.0030951
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0028095, 0.0028278

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016810, upper bound: 0.0016394
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016893, upper bound: 0.0016211
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0013975, 0.0013878
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0002019, 0.0002005
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007673, 0.0007727
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007935, 0.0007991
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008651, 0.0008590
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0008130, 0.0008187
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0032255, 0.0032482
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0044238, 0.0043929
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0031162, 0.0030945
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0028089, 0.0028287

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016464, upper bound: 0.0016016
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016541, upper bound: 0.0015932
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012866, 0.0012689
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001859, 0.0001833
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007015, 0.0007113
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007255, 0.0007357
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007964, 0.0007854
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007433, 0.0007537
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029492, 0.0029904
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0040727, 0.0040165
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028689, 0.0028293
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025683, 0.0026042

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015764, upper bound: 0.0015624
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016102, upper bound: 0.0015524
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012842, 0.0012704
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001855, 0.0001835
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007024, 0.0007100
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007264, 0.0007343
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007950, 0.0007864
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007442, 0.0007523
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029528, 0.0029849
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0040652, 0.0040215
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028636, 0.0028328
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025715, 0.0025994

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016571, upper bound: 0.0015929
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016571, upper bound: 0.0015929
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0011498, 0.0011399
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001661, 0.0001647
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0006302, 0.0006357
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006518, 0.0006575
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007118, 0.0007056
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006677, 0.0006736
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0026494, 0.0026725
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0036397, 0.0036082
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0025639, 0.0025417
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0023072, 0.0023273

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015330, upper bound: 0.0015206
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015369, upper bound: 0.0015204
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0011502, 0.0011397
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001662, 0.0001647
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0006301, 0.0006359
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006517, 0.0006577
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007120, 0.0007055
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006677, 0.0006738
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0026491, 0.0026734
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0036410, 0.0036078
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0025648, 0.0025414
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0023069, 0.0023281

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015330, upper bound: 0.0015206
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015369, upper bound: 0.0015204
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0010417, 0.0010279
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001505, 0.0001485
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0005683, 0.0005759
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0005878, 0.0005956
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0006448, 0.0006363
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006021, 0.0006102
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0023891, 0.0024211
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0032973, 0.0032537
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0023227, 0.0022920
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0020805, 0.0021084

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014989, upper bound: 0.0014871
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015331, upper bound: 0.0014757
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0010412, 0.0010295
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001504, 0.0001487
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0005692, 0.0005757
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0005887, 0.0005954
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0006445, 0.0006373
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006031, 0.0006099
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0023928, 0.0024201
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0032960, 0.0032588
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0023217, 0.0022956
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0020838, 0.0021075

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015076, upper bound: 0.0014801
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015379, upper bound: 0.0014667
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012771, 0.0012624
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001845, 0.0001824
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0006979, 0.0007061
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007218, 0.0007302
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007905, 0.0007814
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007395, 0.0007481
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029342, 0.0029683
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0040425, 0.0039961
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028476, 0.0028149
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025552, 0.0025849

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015228, upper bound: 0.0015032
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015228, upper bound: 0.0015032
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012750, 0.0012654
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001842, 0.0001828
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0006996, 0.0007049
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007236, 0.0007291
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007893, 0.0007833
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007413, 0.0007469
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029411, 0.0029636
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0040361, 0.0040056
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028431, 0.0028216
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025613, 0.0025808

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015839, upper bound: 0.0015533
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015759, upper bound: 0.0015635
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012828, 0.0012585
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001853, 0.0001818
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0006958, 0.0007092
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007196, 0.0007335
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007941, 0.0007790
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007372, 0.0007514
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029250, 0.0029815
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0040606, 0.0039836
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028603, 0.0028061
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025472, 0.0025964

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015255, upper bound: 0.0015030
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015255, upper bound: 0.0015030
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012808, 0.0012616
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001850, 0.0001823
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0006975, 0.0007081
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007214, 0.0007324
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007928, 0.0007810
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007390, 0.0007503
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029323, 0.0029769
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0040543, 0.0039936
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028559, 0.0028132
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025536, 0.0025924

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016135, upper bound: 0.0015414
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016049, upper bound: 0.0015514
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012961, 0.0012728
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001873, 0.0001839
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007037, 0.0007166
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007278, 0.0007411
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008023, 0.0007879
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007456, 0.0007593
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029583, 0.0030126
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0041029, 0.0040289
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028901, 0.0028380
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025762, 0.0026235

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016521, upper bound: 0.0016016
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016425, upper bound: 0.0016108
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012966, 0.0012726
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001873, 0.0001839
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007036, 0.0007169
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007277, 0.0007414
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008026, 0.0007878
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007455, 0.0007596
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029580, 0.0030137
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0041045, 0.0040285
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028913, 0.0028378
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025759, 0.0026245

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015269, upper bound: 0.0015026
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015313, upper bound: 0.0015023
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012933, 0.0012743
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001868, 0.0001841
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007045, 0.0007150
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007286, 0.0007395
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008006, 0.0007888
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007465, 0.0007576
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029617, 0.0030060
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0040939, 0.0040336
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028839, 0.0028414
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025792, 0.0026178

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015292, upper bound: 0.0015014
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015322, upper bound: 0.0015013
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012932, 0.0012741
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001868, 0.0001841
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007044, 0.0007150
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007285, 0.0007395
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008005, 0.0007887
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007464, 0.0007576
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029614, 0.0030058
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0040936, 0.0040331
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028836, 0.0028410
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025789, 0.0026176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016029, upper bound: 0.0015609
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016115, upper bound: 0.0015496
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0011848, 0.0011741
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001712, 0.0001696
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0006491, 0.0006551
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006714, 0.0006775
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007334, 0.0007268
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006878, 0.0006941
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0027290, 0.0027539
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0037505, 0.0037166
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0026419, 0.0026181
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0023765, 0.0023982

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016040, upper bound: 0.0016046
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016383, upper bound: 0.0015901
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0011857, 0.0011734
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001713, 0.0001695
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0006487, 0.0006555
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006710, 0.0006780
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007339, 0.0007263
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006874, 0.0006946
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0027273, 0.0027558
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0037532, 0.0037143
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0026438, 0.0026164
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0023750, 0.0023999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016132, upper bound: 0.0015821
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016472, upper bound: 0.0015699
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0011883, 0.0011707
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001717, 0.0001691
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0006473, 0.0006570
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006694, 0.0006795
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007356, 0.0007247
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006858, 0.0006961
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0027211, 0.0027620
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0037616, 0.0037059
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0026498, 0.0026105
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0023697, 0.0024053

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016810, upper bound: 0.0016468
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016810, upper bound: 0.0016468
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0011899, 0.0011701
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001719, 0.0001691
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0006469, 0.0006579
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006691, 0.0006804
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007366, 0.0007243
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006855, 0.0006970
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0027197, 0.0027657
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0037666, 0.0037040
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0026533, 0.0026092
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0023685, 0.0024085

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016022, upper bound: 0.0015508
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016111, upper bound: 0.0015380
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0011881, 0.0011703
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001716, 0.0001691
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0006470, 0.0006569
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006692, 0.0006794
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007355, 0.0007245
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006856, 0.0006960
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0027202, 0.0027615
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0037609, 0.0037046
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0026492, 0.0026096
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0023689, 0.0024048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016072, upper bound: 0.0016045
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016431, upper bound: 0.0015900
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0011919, 0.0011675
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001722, 0.0001687
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0006455, 0.0006590
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006676, 0.0006815
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007378, 0.0007227
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006839, 0.0006982
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0027135, 0.0027703
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0037729, 0.0036956
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0026577, 0.0026033
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0023631, 0.0024125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015997, upper bound: 0.0015717
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016059, upper bound: 0.0015617
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0011891, 0.0011697
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001718, 0.0001690
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0006467, 0.0006574
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006689, 0.0006800
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007361, 0.0007241
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006852, 0.0006966
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0027188, 0.0027639
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0037642, 0.0037027
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0026516, 0.0026083
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0023676, 0.0024069

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016944, upper bound: 0.0016228
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016944, upper bound: 0.0016228
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0011936, 0.0011668
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001724, 0.0001686
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0006451, 0.0006599
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006672, 0.0006825
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007388, 0.0007222
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006835, 0.0006992
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0027119, 0.0027742
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0037782, 0.0036934
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0026614, 0.0026017
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0023616, 0.0024159

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016160, upper bound: 0.0015822
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016525, upper bound: 0.0015686
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0013984, 0.0013890
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0002020, 0.0002007
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007679, 0.0007731
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007942, 0.0007996
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008656, 0.0008598
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0008137, 0.0008192
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0032284, 0.0032502
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0044265, 0.0043969
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0031181, 0.0030972
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0028115, 0.0028304

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016119, upper bound: 0.0015933
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016159, upper bound: 0.0015931
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0014015, 0.0013855
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0002025, 0.0002002
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007660, 0.0007748
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007922, 0.0008014
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008675, 0.0008576
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0008116, 0.0008210
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0032203, 0.0032574
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0044363, 0.0043857
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0031251, 0.0030894
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0028044, 0.0028367

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016166, upper bound: 0.0015926
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016212, upper bound: 0.0015922
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012728, 0.0012515
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001839, 0.0001808
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0006919, 0.0007037
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007156, 0.0007278
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007879, 0.0007747
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007331, 0.0007456
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029088, 0.0029583
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0040290, 0.0039615
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028381, 0.0027906
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025331, 0.0025762

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015561, upper bound: 0.0015440
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015723, upper bound: 0.0015305
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012714, 0.0012539
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001837, 0.0001811
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0006932, 0.0007029
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007170, 0.0007270
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007870, 0.0007762
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007345, 0.0007448
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029143, 0.0029551
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0040246, 0.0039690
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028350, 0.0027959
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025379, 0.0025734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014901, upper bound: 0.0014668
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014914, upper bound: 0.0014668
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0011750, 0.0011627
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001697, 0.0001680
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0006428, 0.0006496
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006648, 0.0006719
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007273, 0.0007197
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006811, 0.0006883
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0027023, 0.0027309
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0037193, 0.0036803
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0026199, 0.0025925
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0023533, 0.0023782

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014768, upper bound: 0.0014630
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014794, upper bound: 0.0014603
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0011770, 0.0011614
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001700, 0.0001678
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0006421, 0.0006508
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006641, 0.0006730
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007286, 0.0007189
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006803, 0.0006895
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0026994, 0.0027358
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0037259, 0.0036763
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0026246, 0.0025897
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0023507, 0.0023824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015461, upper bound: 0.0015100
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015461, upper bound: 0.0015108
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0013679, 0.0013490
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001976, 0.0001949
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007458, 0.0007563
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007714, 0.0007822
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008468, 0.0008351
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007902, 0.0008013
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0031354, 0.0031795
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0043301, 0.0042702
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0030502, 0.0030080
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0027305, 0.0027688

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014892, upper bound: 0.0014670
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014914, upper bound: 0.0014657
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0013717, 0.0013464
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001982, 0.0001945
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007444, 0.0007584
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007699, 0.0007843
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0008491, 0.0008334
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007887, 0.0008035
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0031294, 0.0031881
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0043419, 0.0042620
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0030585, 0.0030022
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0027252, 0.0027763

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014892, upper bound: 0.0014680
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014914, upper bound: 0.0014668
time: 1.05 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.47 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015521, upper bound: 0.0016080
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015530, upper bound: 0.0016049
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015677, upper bound: 0.0015772
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015680, upper bound: 0.0015759
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015380, upper bound: 0.0016129
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015617, upper bound: 0.0016059
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015963, upper bound: 0.0016541
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015963, upper bound: 0.0016541
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0016042, upper bound: 0.0016479
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015931, upper bound: 0.0016571
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015020, upper bound: 0.0015281
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015022, upper bound: 0.0015269
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0014666, upper bound: 0.0014914
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0014666, upper bound: 0.0014901
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015931, upper bound: 0.0016571
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015934, upper bound: 0.0016541
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015661, upper bound: 0.0016096
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015785, upper bound: 0.0015935
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015929, upper bound: 0.0016426
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0016108, upper bound: 0.0016425
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015663, upper bound: 0.0016023
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015785, upper bound: 0.0015859
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015309, upper bound: 0.0015519
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015315, upper bound: 0.0015449
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015244, upper bound: 0.0015630
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015244, upper bound: 0.0015630
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015248, upper bound: 0.0015562
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015248, upper bound: 0.0015562
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015508, upper bound: 0.0016022
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015459, upper bound: 0.0016028
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015318, upper bound: 0.0015732
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015457, upper bound: 0.0015362
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015672, upper bound: 0.0016009
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015507, upper bound: 0.0016069
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015089, upper bound: 0.0015740
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015219, upper bound: 0.0015376
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015594, upper bound: 0.0016015
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015594, upper bound: 0.0016015
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015101, upper bound: 0.0015691
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015224, upper bound: 0.0015335
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0016810, upper bound: 0.0016394
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0016893, upper bound: 0.0016211
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0016464, upper bound: 0.0016016
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0016541, upper bound: 0.0015932
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015764, upper bound: 0.0015624
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0016102, upper bound: 0.0015524
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0016571, upper bound: 0.0015929
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0016571, upper bound: 0.0015929
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015330, upper bound: 0.0015206
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015369, upper bound: 0.0015204
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015330, upper bound: 0.0015206
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015369, upper bound: 0.0015204
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0014989, upper bound: 0.0014871
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015331, upper bound: 0.0014757
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015076, upper bound: 0.0014801
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015379, upper bound: 0.0014667
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015228, upper bound: 0.0015032
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015228, upper bound: 0.0015032
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015839, upper bound: 0.0015533
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015759, upper bound: 0.0015635
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015255, upper bound: 0.0015030
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015255, upper bound: 0.0015030
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0016135, upper bound: 0.0015414
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0016049, upper bound: 0.0015514
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0016521, upper bound: 0.0016016
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0016425, upper bound: 0.0016108
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015269, upper bound: 0.0015026
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015313, upper bound: 0.0015023
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015292, upper bound: 0.0015014
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015322, upper bound: 0.0015013
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0016029, upper bound: 0.0015609
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0016115, upper bound: 0.0015496
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0016040, upper bound: 0.0016046
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0016383, upper bound: 0.0015901
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0016132, upper bound: 0.0015821
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0016472, upper bound: 0.0015699
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0016810, upper bound: 0.0016468
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0016810, upper bound: 0.0016468
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0016022, upper bound: 0.0015508
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0016111, upper bound: 0.0015380
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0016072, upper bound: 0.0016045
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0016431, upper bound: 0.0015900
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015997, upper bound: 0.0015717
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0016059, upper bound: 0.0015617
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0016944, upper bound: 0.0016228
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0016944, upper bound: 0.0016228
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0016160, upper bound: 0.0015822
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0016525, upper bound: 0.0015686
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0016119, upper bound: 0.0015933
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0016159, upper bound: 0.0015931
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0016166, upper bound: 0.0015926
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0016212, upper bound: 0.0015922
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015561, upper bound: 0.0015440
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015723, upper bound: 0.0015305
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0014901, upper bound: 0.0014668
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0014914, upper bound: 0.0014668
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0014768, upper bound: 0.0014630
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0014794, upper bound: 0.0014603
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015461, upper bound: 0.0015100
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0015461, upper bound: 0.0015108
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0014892, upper bound: 0.0014670
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0014914, upper bound: 0.0014657
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0014892, upper bound: 0.0014680
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 8, lower bound: -0.0014914, upper bound: 0.0014668

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012489, 0.0012745
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001804, 0.0001841
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007046, 0.0006905
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007288, 0.0007141
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007731, 0.0007889
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007466, 0.0007316
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029623, 0.0029029
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0039534, 0.0040344
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0027849, 0.0028419
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025797, 0.0025279

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014574, upper bound: 0.0014826
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014574, upper bound: 0.0014826
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012520, 0.0012721
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001809, 0.0001838
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007033, 0.0006922
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007274, 0.0007159
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007750, 0.0007875
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007452, 0.0007334
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029568, 0.0029101
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0039633, 0.0040269
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0027918, 0.0028367
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025749, 0.0025342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014579, upper bound: 0.0014784
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014579, upper bound: 0.0014784
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012525, 0.0012688
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001809, 0.0001833
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007015, 0.0006925
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007255, 0.0007162
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007753, 0.0007854
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007433, 0.0007337
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029490, 0.0029111
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0039647, 0.0040163
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0027928, 0.0028292
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025681, 0.0025352

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014574, upper bound: 0.0014812
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014574, upper bound: 0.0014812
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012556, 0.0012665
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001814, 0.0001830
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007002, 0.0006942
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007242, 0.0007180
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007772, 0.0007840
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007419, 0.0007355
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029436, 0.0029184
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0039746, 0.0040089
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0027998, 0.0028240
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025634, 0.0025415

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014579, upper bound: 0.0014778
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014579, upper bound: 0.0014778
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0010482, 0.0010736
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001514, 0.0001551
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0005936, 0.0005795
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006139, 0.0005994
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0006489, 0.0006646
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006289, 0.0006141
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0024953, 0.0024364
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0033181, 0.0033984
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0023374, 0.0023939
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0021730, 0.0021217

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014981, upper bound: 0.0015744
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014981, upper bound: 0.0015744
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0010489, 0.0010710
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001515, 0.0001547
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0005921, 0.0005799
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006124, 0.0005998
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0006493, 0.0006630
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006274, 0.0006145
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0024893, 0.0024380
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0033204, 0.0033902
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0023389, 0.0023881
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0021678, 0.0021231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014691, upper bound: 0.0015245
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014824, upper bound: 0.0014947
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012663, 0.0012834
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001829, 0.0001854
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007095, 0.0007001
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007338, 0.0007241
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007838, 0.0007944
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007518, 0.0007418
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029829, 0.0029432
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0040084, 0.0040624
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028236, 0.0028617
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025976, 0.0025631

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014511, upper bound: 0.0014854
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014511, upper bound: 0.0014845
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012664, 0.0012834
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001830, 0.0001854
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007095, 0.0007002
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007338, 0.0007242
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007839, 0.0007944
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007518, 0.0007419
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029829, 0.0029436
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0040089, 0.0040624
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028239, 0.0028616
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025976, 0.0025634

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014982, upper bound: 0.0015720
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015227, upper bound: 0.0015630
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012648, 0.0012823
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001827, 0.0001852
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007089, 0.0006993
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007332, 0.0007232
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007829, 0.0007937
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007511, 0.0007409
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029803, 0.0029397
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0040036, 0.0040589
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028202, 0.0028592
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025954, 0.0025600

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014571, upper bound: 0.0014826
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014571, upper bound: 0.0014812
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012631, 0.0012833
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001825, 0.0001854
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007095, 0.0006984
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007338, 0.0007223
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007819, 0.0007944
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007518, 0.0007399
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029828, 0.0029359
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0039984, 0.0040623
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028166, 0.0028616
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025976, 0.0025567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014509, upper bound: 0.0014877
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014509, upper bound: 0.0014863
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012575, 0.0012768
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001817, 0.0001845
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007059, 0.0006952
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007301, 0.0007190
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007784, 0.0007903
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007479, 0.0007366
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029675, 0.0029227
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0039804, 0.0040415
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028039, 0.0028469
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025843, 0.0025452

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014103, upper bound: 0.0014441
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014193, upper bound: 0.0014360
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012612, 0.0012709
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001822, 0.0001836
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007027, 0.0006973
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007267, 0.0007212
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007807, 0.0007867
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007445, 0.0007388
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029540, 0.0029315
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0039924, 0.0040231
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028123, 0.0028340
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025725, 0.0025529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014577, upper bound: 0.0014778
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014511, upper bound: 0.0014845
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012340, 0.0012549
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001783, 0.0001813
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0006938, 0.0006822
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007175, 0.0007056
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007638, 0.0007768
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007351, 0.0007229
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029166, 0.0028681
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0039061, 0.0039722
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0027515, 0.0027981
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025399, 0.0024976

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014571, upper bound: 0.0014826
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014577, upper bound: 0.0014784
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012377, 0.0012491
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001788, 0.0001805
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0006906, 0.0006843
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007143, 0.0007077
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007661, 0.0007732
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007317, 0.0007250
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029033, 0.0028767
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0039178, 0.0039540
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0027598, 0.0027853
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025283, 0.0025051

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013746, upper bound: 0.0014097
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013845, upper bound: 0.0013978
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012633, 0.0012835
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001825, 0.0001854
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007096, 0.0006984
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007339, 0.0007224
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007820, 0.0007945
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007518, 0.0007400
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029831, 0.0029362
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0039989, 0.0040627
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028169, 0.0028619
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025978, 0.0025570

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014971, upper bound: 0.0015744
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015169, upper bound: 0.0015660
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012668, 0.0012810
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001830, 0.0001851
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007082, 0.0007004
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007325, 0.0007244
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007842, 0.0007930
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007504, 0.0007421
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029775, 0.0029444
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0040100, 0.0040550
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028247, 0.0028564
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025929, 0.0025641

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014974, upper bound: 0.0015720
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014971, upper bound: 0.0015630
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0010620, 0.0010895
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001534, 0.0001574
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0006024, 0.0005871
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006230, 0.0006072
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0006574, 0.0006744
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006383, 0.0006221
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0025324, 0.0024683
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0033616, 0.0034489
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0023680, 0.0024295
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0022053, 0.0021495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015221, upper bound: 0.0015642
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015187, upper bound: 0.0015642
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0010660, 0.0010872
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001540, 0.0001571
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0006011, 0.0005894
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006217, 0.0006095
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0006599, 0.0006730
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006369, 0.0006245
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0025269, 0.0024777
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0033744, 0.0034414
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0023770, 0.0024242
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0022005, 0.0021577

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014519, upper bound: 0.0014657
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014522, upper bound: 0.0014583
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012625, 0.0012872
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001824, 0.0001860
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007116, 0.0006980
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007360, 0.0007219
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007815, 0.0007968
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007540, 0.0007396
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029917, 0.0029343
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0039963, 0.0040745
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028151, 0.0028702
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0026053, 0.0025553

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015221, upper bound: 0.0015642
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015356, upper bound: 0.0015478
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012627, 0.0012852
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001824, 0.0001857
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007105, 0.0006981
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007349, 0.0007220
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007817, 0.0007955
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007529, 0.0007397
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029871, 0.0029350
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0039972, 0.0040682
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028157, 0.0028657
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0026013, 0.0025559

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014584, upper bound: 0.0014805
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014584, upper bound: 0.0014785
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0010640, 0.0010859
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001537, 0.0001569
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0006004, 0.0005883
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006209, 0.0006084
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0006587, 0.0006722
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006361, 0.0006233
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0025239, 0.0024731
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0033682, 0.0034374
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0023726, 0.0024213
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0021979, 0.0021537

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014980, upper bound: 0.0015570
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015194, upper bound: 0.0015570
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0010688, 0.0010838
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001544, 0.0001566
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0005992, 0.0005909
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006197, 0.0006111
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0006616, 0.0006709
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006349, 0.0006261
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0025191, 0.0024841
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0033831, 0.0034308
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0023831, 0.0024167
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0021937, 0.0021632

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014528, upper bound: 0.0014601
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014531, upper bound: 0.0014530
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012633, 0.0012878
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001825, 0.0001861
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007120, 0.0006984
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007364, 0.0007223
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007820, 0.0007972
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007544, 0.0007400
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029932, 0.0029361
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0039988, 0.0040765
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028168, 0.0028716
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0026066, 0.0025569

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014590, upper bound: 0.0014735
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014589, upper bound: 0.0014735
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012672, 0.0012822
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001831, 0.0001852
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007089, 0.0007006
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007332, 0.0007246
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007844, 0.0007937
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007511, 0.0007423
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029801, 0.0029453
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0040112, 0.0040587
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028256, 0.0028590
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0025952, 0.0025649

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014410, upper bound: 0.0014673
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014531, upper bound: 0.0014530
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012601, 0.0012925
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001820, 0.0001867
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007146, 0.0006967
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007390, 0.0007205
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007800, 0.0008000
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007571, 0.0007381
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0030040, 0.0029287
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0039887, 0.0040912
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028097, 0.0028819
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0026160, 0.0025505

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014319, upper bound: 0.0014820
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014497, upper bound: 0.0014721
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012602, 0.0012919
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001821, 0.0001866
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007143, 0.0006967
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007387, 0.0007206
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007801, 0.0007997
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007568, 0.0007382
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0030028, 0.0029290
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0039891, 0.0040895
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028100, 0.0028807
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0026149, 0.0025507

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014536, upper bound: 0.0014861
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014536, upper bound: 0.0014861
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012635, 0.0012870
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001825, 0.0001859
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007116, 0.0006985
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007359, 0.0007225
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007821, 0.0007967
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007539, 0.0007401
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029914, 0.0029367
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0039995, 0.0040741
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028173, 0.0028699
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0026051, 0.0025574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014324, upper bound: 0.0014752
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014499, upper bound: 0.0014661
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0012636, 0.0012864
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001825, 0.0001858
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0007112, 0.0006986
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0007356, 0.0007225
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0007822, 0.0007963
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0007536, 0.0007402
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0029900, 0.0029369
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0039998, 0.0040721
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0028175, 0.0028685
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0026038, 0.0025576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014324, upper bound: 0.0014752
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014499, upper bound: 0.0014661
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0010486, 0.0010714
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001515, 0.0001548
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0005923, 0.0005797
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006126, 0.0005996
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0006491, 0.0006632
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006276, 0.0006143
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0024902, 0.0024372
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0033192, 0.0033914
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0023381, 0.0023890
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0021685, 0.0021224

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015085, upper bound: 0.0015649
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014971, upper bound: 0.0015649
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0010488, 0.0010693
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001515, 0.0001545
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0005912, 0.0005798
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006115, 0.0005997
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0006492, 0.0006619
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006264, 0.0006144
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0024854, 0.0024376
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0033198, 0.0033850
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0023386, 0.0023844
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0021644, 0.0021228

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014971, upper bound: 0.0015649
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015066, upper bound: 0.0015649
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0010529, 0.0010736
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001521, 0.0001551
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0005936, 0.0005821
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006139, 0.0006021
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0006518, 0.0006646
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006289, 0.0006168
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0024954, 0.0024473
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0033331, 0.0033985
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0023479, 0.0023940
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0021731, 0.0021312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014504, upper bound: 0.0014677
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014504, upper bound: 0.0014677
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0010560, 0.0010681
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001526, 0.0001543
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0005905, 0.0005838
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006107, 0.0006038
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0006537, 0.0006611
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006257, 0.0006186
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0024825, 0.0024544
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0033426, 0.0033809
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0023546, 0.0023816
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0021618, 0.0021374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014902, upper bound: 0.0014823
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014872, upper bound: 0.0014823
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0010507, 0.0010705
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001518, 0.0001547
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0005918, 0.0005809
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006121, 0.0006008
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0006504, 0.0006626
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006271, 0.0006155
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0024881, 0.0024421
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0033259, 0.0033886
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0023428, 0.0023870
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0021668, 0.0021267

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014702, upper bound: 0.0015249
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014823, upper bound: 0.0014911
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0010490, 0.0010706
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001516, 0.0001547
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0005919, 0.0005800
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006122, 0.0005999
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0006494, 0.0006627
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006272, 0.0006145
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0024884, 0.0024383
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0033207, 0.0033890
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0023392, 0.0023873
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0021670, 0.0021233

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014590, upper bound: 0.0015285
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014700, upper bound: 0.0014949
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0010437, 0.0010689
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001508, 0.0001544
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0005910, 0.0005770
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006112, 0.0005968
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0006461, 0.0006617
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006262, 0.0006114
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0024845, 0.0024258
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0033038, 0.0033837
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0023272, 0.0023836
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0021636, 0.0021125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014666, upper bound: 0.0015249
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014578, upper bound: 0.0015285
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0010471, 0.0010622
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001513, 0.0001535
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0005873, 0.0005789
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006074, 0.0005987
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0006482, 0.0006575
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006223, 0.0006134
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0024689, 0.0024337
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0033145, 0.0033625
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0023348, 0.0023686
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0021500, 0.0021194

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014091, upper bound: 0.0014432
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014091, upper bound: 0.0014432
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0010614, 0.0010768
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001533, 0.0001556
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0005953, 0.0005868
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006157, 0.0006069
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0006570, 0.0006666
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006308, 0.0006218
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0025028, 0.0024671
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0033599, 0.0034086
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0023668, 0.0024011
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0021796, 0.0021484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015221, upper bound: 0.0015570
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015085, upper bound: 0.0015649
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0023945, 0.0049173, 0.0023945, 0.0049173, -0.0010617, 0.0010766
1: 0.0016682, 0.0020327, 0.0016682, 0.0020327, -0.0001534, 0.0001555
2: 0.0116412, 0.0130360, 0.0116412, 0.0130360, -0.0005952, 0.0005870
3: -0.0026406, -0.0011980, -0.0026406, -0.0011980, -0.0006156, 0.0006071
4: -0.0027400, -0.0011784, -0.0027400, -0.0011784, -0.0006572, 0.0006664
5: 0.0052285, 0.0067063, 0.0052285, 0.0067063, -0.0006306, 0.0006219
6: -0.0015553, 0.0043085, -0.0015553, 0.0043085, -0.0025022, 0.0024676
7: -0.0084245, -0.0004386, -0.0084245, -0.0004386, -0.0033607, 0.0034078
8: 0.9832795, 0.9889049, 0.9832795, 0.9889049, -0.0023674, 0.0024005
9: -0.0058159, -0.0007095, -0.0058159, -0.0007095, -0.0021790, 0.0021489

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014097, upper bound: 0.0014411
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014097, upper bound: 0.0014392
time: 0.87 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.25 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014574, upper bound: 0.0014826
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014574, upper bound: 0.0014826
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014579, upper bound: 0.0014784
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014579, upper bound: 0.0014784
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014574, upper bound: 0.0014812
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014574, upper bound: 0.0014812
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014579, upper bound: 0.0014778
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014579, upper bound: 0.0014778
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014981, upper bound: 0.0015744
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014981, upper bound: 0.0015744
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014691, upper bound: 0.0015245
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014824, upper bound: 0.0014947
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014511, upper bound: 0.0014854
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014511, upper bound: 0.0014845
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014982, upper bound: 0.0015720
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0015227, upper bound: 0.0015630
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014571, upper bound: 0.0014826
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014571, upper bound: 0.0014812
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014509, upper bound: 0.0014877
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014509, upper bound: 0.0014863
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014103, upper bound: 0.0014441
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014193, upper bound: 0.0014360
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014577, upper bound: 0.0014778
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014511, upper bound: 0.0014845
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014571, upper bound: 0.0014826
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014577, upper bound: 0.0014784
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0013746, upper bound: 0.0014097
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0013845, upper bound: 0.0013978
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014971, upper bound: 0.0015744
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0015169, upper bound: 0.0015660
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014974, upper bound: 0.0015720
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014971, upper bound: 0.0015630
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0015221, upper bound: 0.0015642
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0015187, upper bound: 0.0015642
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014519, upper bound: 0.0014657
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014522, upper bound: 0.0014583
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0015221, upper bound: 0.0015642
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0015356, upper bound: 0.0015478
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014584, upper bound: 0.0014805
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014584, upper bound: 0.0014785
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014980, upper bound: 0.0015570
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0015194, upper bound: 0.0015570
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014528, upper bound: 0.0014601
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014531, upper bound: 0.0014530
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014590, upper bound: 0.0014735
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014589, upper bound: 0.0014735
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014410, upper bound: 0.0014673
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014531, upper bound: 0.0014530
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014319, upper bound: 0.0014820
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014497, upper bound: 0.0014721
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014536, upper bound: 0.0014861
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014536, upper bound: 0.0014861
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014324, upper bound: 0.0014752
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014499, upper bound: 0.0014661
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014324, upper bound: 0.0014752
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014499, upper bound: 0.0014661
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0015085, upper bound: 0.0015649
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014971, upper bound: 0.0015649
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014971, upper bound: 0.0015649
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0015066, upper bound: 0.0015649
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014504, upper bound: 0.0014677
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014504, upper bound: 0.0014677
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014902, upper bound: 0.0014823
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014872, upper bound: 0.0014823
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014702, upper bound: 0.0015249
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014823, upper bound: 0.0014911
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014590, upper bound: 0.0015285
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014700, upper bound: 0.0014949
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014666, upper bound: 0.0015249
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014578, upper bound: 0.0015285
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014091, upper bound: 0.0014432
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014091, upper bound: 0.0014432
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0015221, upper bound: 0.0015570
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0015085, upper bound: 0.0015649
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014097, upper bound: 0.0014411
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.25
Output dim: 8, lower bound: -0.0014097, upper bound: 0.0014392
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0015101, upper bound: 0.0015691
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0015224, upper bound: 0.0015335
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0016810, upper bound: 0.0016394
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0016893, upper bound: 0.0016211
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0016464, upper bound: 0.0016016
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0016541, upper bound: 0.0015932
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0015764, upper bound: 0.0015624
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0016102, upper bound: 0.0015524
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0016571, upper bound: 0.0015929
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0016571, upper bound: 0.0015929
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0015330, upper bound: 0.0015206
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0015369, upper bound: 0.0015204
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0015330, upper bound: 0.0015206
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0015369, upper bound: 0.0015204
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0014989, upper bound: 0.0014871
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0015331, upper bound: 0.0014757
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0015076, upper bound: 0.0014801
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0015379, upper bound: 0.0014667
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0015228, upper bound: 0.0015032
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0015228, upper bound: 0.0015032
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0015839, upper bound: 0.0015533
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0015759, upper bound: 0.0015635
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0015255, upper bound: 0.0015030
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0015255, upper bound: 0.0015030
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0016135, upper bound: 0.0015414
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0016049, upper bound: 0.0015514
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0016521, upper bound: 0.0016016
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0016425, upper bound: 0.0016108
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0015269, upper bound: 0.0015026
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0015313, upper bound: 0.0015023
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0015292, upper bound: 0.0015014
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0015322, upper bound: 0.0015013
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0016029, upper bound: 0.0015609
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0016115, upper bound: 0.0015496
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0016040, upper bound: 0.0016046
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0016383, upper bound: 0.0015901
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0016132, upper bound: 0.0015821
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0016472, upper bound: 0.0015699
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0016810, upper bound: 0.0016468
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0016810, upper bound: 0.0016468
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0016022, upper bound: 0.0015508
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0016111, upper bound: 0.0015380
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0016072, upper bound: 0.0016045
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0016431, upper bound: 0.0015900
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0015997, upper bound: 0.0015717
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0016059, upper bound: 0.0015617
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0016944, upper bound: 0.0016228
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0016944, upper bound: 0.0016228
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0016160, upper bound: 0.0015822
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0016525, upper bound: 0.0015686
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0016119, upper bound: 0.0015933
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0016159, upper bound: 0.0015931
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0016166, upper bound: 0.0015926
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0016212, upper bound: 0.0015922
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0015561, upper bound: 0.0015440
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0015723, upper bound: 0.0015305
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0014901, upper bound: 0.0014668
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0014914, upper bound: 0.0014668
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0014768, upper bound: 0.0014630
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0014794, upper bound: 0.0014603
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0015461, upper bound: 0.0015100
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0015461, upper bound: 0.0015108
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0014892, upper bound: 0.0014670
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0014914, upper bound: 0.0014657
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0014892, upper bound: 0.0014680
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 8, lower bound: -0.0014914, upper bound: 0.0014668

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 2.97 + 598.22 = 601.19 seconds
