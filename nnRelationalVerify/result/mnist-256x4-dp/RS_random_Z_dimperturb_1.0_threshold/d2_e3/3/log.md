## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00157248


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0062809, 0.0062809)
1: (-0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0017708, 0.0017708)
2: (-0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0130656, 0.0130656)
3: (0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0017290, 0.0017290)
4: (-0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0097644, 0.0097644)
5: (0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0027128, 0.0027128)
6: (0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0024624, 0.0024624)
7: (-0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0091894, 0.0091894)
8: (-0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0071521, 0.0071521)
9: (-0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0006171, 0.0006171)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.32 + 2.92 = 4.24 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0017475, upper bound: 0.0017472

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0017266, upper bound: 0.0017183
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0017180, upper bound: 0.0017268
time: 2.09 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 3.81 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 3.81
Output dim: 5, lower bound: -0.0017266, upper bound: 0.0017183
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 3.81
Output dim: 5, lower bound: -0.0017180, upper bound: 0.0017268

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0061847, 0.0062040
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0017437, 0.0017491
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0128654, 0.0129056
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0017025, 0.0017079
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0096449, 0.0096148
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0026796, 0.0026713
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0024323, 0.0024247
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0090769, 0.0090486
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0070425, 0.0070646
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0006095, 0.0006076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0011580, upper bound: 0.0011576
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0011580, upper bound: 0.0011576
time: 0.88 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0062040, 0.0061847
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0017491, 0.0017437
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0129056, 0.0128654
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0017079, 0.0017025
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0096148, 0.0096449
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0026713, 0.0026796
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0024247, 0.0024323
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0090486, 0.0090769
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0070646, 0.0070425
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0006076, 0.0006095

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016970, upper bound: 0.0017014
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016915, upper bound: 0.0017052
time: 2.08 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 5.02 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 5.02
Output dim: 5, lower bound: -0.0011580, upper bound: 0.0011576
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 5.02
Output dim: 5, lower bound: -0.0011580, upper bound: 0.0011576
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 5.02
Output dim: 5, lower bound: -0.0016970, upper bound: 0.0017014
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 5.02
Output dim: 5, lower bound: -0.0016915, upper bound: 0.0017052

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0061321, 0.0061184
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0017289, 0.0017250
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0127560, 0.0127275
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016881, 0.0016843
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0095117, 0.0095330
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0026426, 0.0026486
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023987, 0.0024041
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0089516, 0.0089717
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0069827, 0.0069670
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0006011, 0.0006024

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016158, upper bound: 0.0016180
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016153, upper bound: 0.0016184
time: 1.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0061361, 0.0061127
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0017300, 0.0017234
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0127644, 0.0127157
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016892, 0.0016827
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0095029, 0.0095393
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0026402, 0.0026503
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023965, 0.0024057
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0089433, 0.0089776
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0069873, 0.0069606
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0006005, 0.0006028

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016456, upper bound: 0.0016605
time: 1.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016461, upper bound: 0.0016598
time: 2.09 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 5.27 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.27
Output dim: 5, lower bound: -0.0016158, upper bound: 0.0016180
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.27
Output dim: 5, lower bound: -0.0016153, upper bound: 0.0016184
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.27
Output dim: 5, lower bound: -0.0016456, upper bound: 0.0016605
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.27
Output dim: 5, lower bound: -0.0016461, upper bound: 0.0016598

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0060717, 0.0060886
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0017118, 0.0017166
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0126304, 0.0126655
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016714, 0.0016761
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0094654, 0.0094392
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0026298, 0.0026225
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023870, 0.0023804
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0089080, 0.0088833
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0069139, 0.0069331
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005982, 0.0005965

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015948, upper bound: 0.0015738
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015714, upper bound: 0.0015966
time: 1.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0061321, 0.0060580
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0017289, 0.0017080
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0127560, 0.0126019
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016881, 0.0016677
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0094179, 0.0095330
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0026166, 0.0026486
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023751, 0.0024041
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0088633, 0.0089717
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0069827, 0.0068983
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005952, 0.0006024

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015944, upper bound: 0.0015738
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015710, upper bound: 0.0015975
time: 1.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0059038, 0.0058925
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016645, 0.0016613
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0122812, 0.0122575
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016252, 0.0016221
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0091605, 0.0091782
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025451, 0.0025500
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023102, 0.0023146
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0086211, 0.0086377
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0067227, 0.0067098
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005789, 0.0005800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015978, upper bound: 0.0016056
time: 1.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015948, upper bound: 0.0016110
time: 1.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0059159, 0.0058806
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016679, 0.0016579
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0123062, 0.0122327
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016285, 0.0016188
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0091420, 0.0091969
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025399, 0.0025552
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023055, 0.0023193
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0086036, 0.0086553
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0067364, 0.0066962
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005777, 0.0005812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016305, upper bound: 0.0016447
time: 1.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016312, upper bound: 0.0016449
time: 1.89 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.96 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.96
Output dim: 5, lower bound: -0.0015948, upper bound: 0.0015738
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.96
Output dim: 5, lower bound: -0.0015714, upper bound: 0.0015966
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.96
Output dim: 5, lower bound: -0.0015944, upper bound: 0.0015738
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.96
Output dim: 5, lower bound: -0.0015710, upper bound: 0.0015975
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.96
Output dim: 5, lower bound: -0.0015978, upper bound: 0.0016056
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.96
Output dim: 5, lower bound: -0.0015948, upper bound: 0.0016110
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.96
Output dim: 5, lower bound: -0.0016305, upper bound: 0.0016447
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.96
Output dim: 5, lower bound: -0.0016312, upper bound: 0.0016449

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0060037, 0.0060414
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016927, 0.0017033
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0124888, 0.0125674
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016527, 0.0016631
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0093921, 0.0093334
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0026094, 0.0025931
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023686, 0.0023537
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0088390, 0.0087837
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0068364, 0.0068794
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005935, 0.0005898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015422, upper bound: 0.0015224
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015422, upper bound: 0.0015233
time: 1.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0060246, 0.0060219
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016986, 0.0016978
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0125324, 0.0125267
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016585, 0.0016577
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0093617, 0.0093659
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0026010, 0.0026021
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023609, 0.0023619
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0088104, 0.0088144
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0068602, 0.0068571
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005916, 0.0005919

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015257, upper bound: 0.0015518
time: 2.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015291, upper bound: 0.0015512
time: 1.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0060646, 0.0060109
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0017099, 0.0016947
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0126157, 0.0125039
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016695, 0.0016547
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0093446, 0.0094282
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025962, 0.0026194
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023566, 0.0023777
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0087943, 0.0088730
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0069059, 0.0068446
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005905, 0.0005958

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015935, upper bound: 0.0015670
time: 1.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015852, upper bound: 0.0015730
time: 1.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0060856, 0.0059907
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0017158, 0.0016890
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0126593, 0.0124619
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016753, 0.0016491
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0093132, 0.0094607
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025875, 0.0026285
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023487, 0.0023859
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0087648, 0.0089036
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0069297, 0.0068216
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005885, 0.0005979

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015250, upper bound: 0.0015480
time: 1.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015250, upper bound: 0.0015477
time: 1.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0058184, 0.0058251
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016404, 0.0016423
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0121035, 0.0121175
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016017, 0.0016036
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0090558, 0.0090454
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025160, 0.0025131
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0022838, 0.0022811
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0085226, 0.0085128
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0066255, 0.0066331
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005723, 0.0005716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013633, upper bound: 0.0013671
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013633, upper bound: 0.0013671
time: 1.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0058365, 0.0058925
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016455, 0.0016613
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0121411, 0.0122575
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016067, 0.0016221
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0091605, 0.0090735
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025451, 0.0025209
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023102, 0.0022882
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0086211, 0.0085392
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0066461, 0.0067098
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005789, 0.0005734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015747, upper bound: 0.0015732
time: 2.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015586, upper bound: 0.0015911
time: 2.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0058732, 0.0058344
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016559, 0.0016449
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0122175, 0.0121367
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016168, 0.0016061
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0090702, 0.0091306
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025200, 0.0025367
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0022874, 0.0023026
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0085361, 0.0085929
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0066879, 0.0066437
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005732, 0.0005770

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015962, upper bound: 0.0016006
time: 1.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015859, upper bound: 0.0016093
time: 1.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0058733, 0.0058379
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016559, 0.0016459
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0122177, 0.0121440
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016168, 0.0016071
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0090757, 0.0091307
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025215, 0.0025368
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0022887, 0.0023026
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0085412, 0.0085930
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0066880, 0.0066476
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005735, 0.0005770

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015614, upper bound: 0.0015700
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015615, upper bound: 0.0015706
time: 1.42 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.05 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 5, lower bound: -0.0015422, upper bound: 0.0015224
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 5, lower bound: -0.0015422, upper bound: 0.0015233
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 5, lower bound: -0.0015257, upper bound: 0.0015518
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 5, lower bound: -0.0015291, upper bound: 0.0015512
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.05
Output dim: 5, lower bound: -0.0015935, upper bound: 0.0015670
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.05
Output dim: 5, lower bound: -0.0015852, upper bound: 0.0015730
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 5, lower bound: -0.0015250, upper bound: 0.0015480
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 5, lower bound: -0.0015250, upper bound: 0.0015477
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 5, lower bound: -0.0013633, upper bound: 0.0013671
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 5, lower bound: -0.0013633, upper bound: 0.0013671
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.05
Output dim: 5, lower bound: -0.0015747, upper bound: 0.0015732
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.05
Output dim: 5, lower bound: -0.0015586, upper bound: 0.0015911
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.05
Output dim: 5, lower bound: -0.0015962, upper bound: 0.0016006
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.05
Output dim: 5, lower bound: -0.0015859, upper bound: 0.0016093
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 5, lower bound: -0.0015614, upper bound: 0.0015700
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 5, lower bound: -0.0015615, upper bound: 0.0015706

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0060249, 0.0059864
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016986, 0.0016878
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0125330, 0.0124529
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016585, 0.0016479
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0093065, 0.0093664
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025856, 0.0026023
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023470, 0.0023621
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0087585, 0.0088148
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0068606, 0.0068167
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005881, 0.0005919

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015825, upper bound: 0.0015500
time: 1.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015751, upper bound: 0.0015557
time: 1.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0060388, 0.0059711
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0017026, 0.0016835
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0125620, 0.0124210
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016624, 0.0016437
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0092827, 0.0093880
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025790, 0.0026083
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023410, 0.0023675
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0087361, 0.0088352
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0068764, 0.0067993
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005866, 0.0005933

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014322, upper bound: 0.0014090
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014322, upper bound: 0.0014090
time: 1.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0057849, 0.0058579
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016310, 0.0016515
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0120337, 0.0121855
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0015925, 0.0016126
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0091067, 0.0089932
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025301, 0.0024986
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0022966, 0.0022680
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0085704, 0.0084637
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0065873, 0.0066704
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005755, 0.0005683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015672, upper bound: 0.0015664
time: 2.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015684, upper bound: 0.0015650
time: 1.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0058045, 0.0058366
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016365, 0.0016455
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0120745, 0.0121412
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0015979, 0.0016067
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0090736, 0.0090237
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025209, 0.0025071
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0022882, 0.0022757
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0085393, 0.0084923
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0066096, 0.0066461
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005734, 0.0005702

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015478, upper bound: 0.0015754
time: 1.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015451, upper bound: 0.0015811
time: 1.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0058307, 0.0058148
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016439, 0.0016394
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0121289, 0.0120960
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016051, 0.0016007
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0090398, 0.0090644
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025115, 0.0025184
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0022797, 0.0022859
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0085075, 0.0085306
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0066394, 0.0066214
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005713, 0.0005728

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015829, upper bound: 0.0015881
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015837, upper bound: 0.0015871
time: 1.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0058549, 0.0057918
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016507, 0.0016329
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0121793, 0.0120482
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016117, 0.0015944
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0090041, 0.0091021
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025016, 0.0025288
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0022707, 0.0022954
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0084738, 0.0085661
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0066670, 0.0065952
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005690, 0.0005752

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015428, upper bound: 0.0015603
time: 1.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015378, upper bound: 0.0015616
time: 1.64 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 6.67 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.67
Output dim: 5, lower bound: -0.0015825, upper bound: 0.0015500
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.67
Output dim: 5, lower bound: -0.0015751, upper bound: 0.0015557
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 6.67
Output dim: 5, lower bound: -0.0014322, upper bound: 0.0014090
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 6.67
Output dim: 5, lower bound: -0.0014322, upper bound: 0.0014090
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 6.67
Output dim: 5, lower bound: -0.0015672, upper bound: 0.0015664
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 6.67
Output dim: 5, lower bound: -0.0015684, upper bound: 0.0015650
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.67
Output dim: 5, lower bound: -0.0015478, upper bound: 0.0015754
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.67
Output dim: 5, lower bound: -0.0015451, upper bound: 0.0015811
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.67
Output dim: 5, lower bound: -0.0015829, upper bound: 0.0015881
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.67
Output dim: 5, lower bound: -0.0015837, upper bound: 0.0015871
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 6.67
Output dim: 5, lower bound: -0.0015428, upper bound: 0.0015603
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 6.67
Output dim: 5, lower bound: -0.0015378, upper bound: 0.0015616

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0059557, 0.0059384
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016791, 0.0016742
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0123890, 0.0123530
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016395, 0.0016347
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0092319, 0.0092588
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025649, 0.0025724
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023281, 0.0023349
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0086882, 0.0087136
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0067818, 0.0067621
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005834, 0.0005851

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0012912, upper bound: 0.0012544
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0012912, upper bound: 0.0012544
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0059771, 0.0059170
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016852, 0.0016682
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0124335, 0.0123085
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016454, 0.0016288
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0091986, 0.0092920
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025556, 0.0025816
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023197, 0.0023433
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0086569, 0.0087448
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0068061, 0.0067377
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005813, 0.0005872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015745, upper bound: 0.0015232
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015141, upper bound: 0.0015555
time: 1.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0057340, 0.0057884
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016166, 0.0016320
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0119279, 0.0120410
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0015785, 0.0015934
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0089987, 0.0089142
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025001, 0.0024766
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0022693, 0.0022480
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0084688, 0.0083892
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0065294, 0.0065912
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005687, 0.0005633

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014968, upper bound: 0.0015247
time: 1.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014982, upper bound: 0.0015236
time: 1.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0057553, 0.0057661
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016226, 0.0016257
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0119723, 0.0119946
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0015843, 0.0015873
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0089641, 0.0089473
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0024905, 0.0024858
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0022606, 0.0022564
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0084362, 0.0084204
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0065536, 0.0065659
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005665, 0.0005654

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015318, upper bound: 0.0015678
time: 1.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015319, upper bound: 0.0015679
time: 1.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0057777, 0.0057635
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016290, 0.0016249
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0120189, 0.0119892
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0015905, 0.0015866
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0089600, 0.0089821
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0024894, 0.0024955
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0022596, 0.0022652
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0084324, 0.0084532
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0065791, 0.0065629
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005662, 0.0005676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015638, upper bound: 0.0015672
time: 2.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015623, upper bound: 0.0015690
time: 1.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0057793, 0.0057636
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016294, 0.0016250
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0120222, 0.0119894
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0015909, 0.0015866
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0089601, 0.0089846
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0024894, 0.0024962
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0022596, 0.0022658
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0084325, 0.0084555
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0065810, 0.0065630
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005662, 0.0005678

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015046, upper bound: 0.0015077
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015034, upper bound: 0.0015086
time: 1.65 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 4.42 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0012912, upper bound: 0.0012544
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0012912, upper bound: 0.0012544
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015745, upper bound: 0.0015232
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015141, upper bound: 0.0015555
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0014968, upper bound: 0.0015247
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0014982, upper bound: 0.0015236
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015318, upper bound: 0.0015678
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015319, upper bound: 0.0015679
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015638, upper bound: 0.0015672
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015623, upper bound: 0.0015690
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015046, upper bound: 0.0015077
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015034, upper bound: 0.0015086

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0059601, 0.0059779
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016804, 0.0016854
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0123983, 0.0124353
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016407, 0.0016456
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0092934, 0.0092657
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025820, 0.0025743
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023437, 0.0023367
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0087461, 0.0087201
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0067869, 0.0068071
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005873, 0.0005855

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013080, upper bound: 0.0012799
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013080, upper bound: 0.0012799
time: 1.40 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 7.90 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 7.90
Output dim: 5, lower bound: -0.0013080, upper bound: 0.0012799
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 7.90
Output dim: 5, lower bound: -0.0013080, upper bound: 0.0012799

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 4.24 + 152.54 = 156.78 seconds
