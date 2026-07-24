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
execution time: IAR + RelationalAnalysis = 1.33 + 2.93 = 4.26 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0017475, upper bound: 0.0017472

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0017114, upper bound: 0.0017033
time: 1.83 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0017034, upper bound: 0.0017116
time: 1.42 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 3.38 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 3.38
Output dim: 5, lower bound: -0.0017114, upper bound: 0.0017033
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 3.38
Output dim: 5, lower bound: -0.0017034, upper bound: 0.0017116

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0062371, 0.0062600
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0017585, 0.0017649
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0129744, 0.0130220
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0017170, 0.0017233
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0097319, 0.0096963
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0027038, 0.0026939
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0024542, 0.0024453
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0091588, 0.0091253
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0071022, 0.0071283
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0006150, 0.0006127

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016619, upper bound: 0.0016632
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016693, upper bound: 0.0016517
time: 1.86 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0062600, 0.0062371
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0017649, 0.0017585
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0130220, 0.0129744
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0017233, 0.0017170
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0096963, 0.0097319
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0026939, 0.0027038
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0024453, 0.0024542
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0091253, 0.0091588
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0071283, 0.0071022
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0006127, 0.0006150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016520, upper bound: 0.0016691
time: 1.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016629, upper bound: 0.0016621
time: 1.80 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 5.09 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 5.09
Output dim: 5, lower bound: -0.0016619, upper bound: 0.0016632
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 5.09
Output dim: 5, lower bound: -0.0016693, upper bound: 0.0016517
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 5.09
Output dim: 5, lower bound: -0.0016520, upper bound: 0.0016691
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 5.09
Output dim: 5, lower bound: -0.0016629, upper bound: 0.0016621

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0058809, 0.0058876
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016580, 0.0016599
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0122334, 0.0122474
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016189, 0.0016208
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0091530, 0.0091425
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025430, 0.0025400
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023082, 0.0023056
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0086140, 0.0086041
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0066966, 0.0067043
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005784, 0.0005777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016407, upper bound: 0.0016411
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016370, upper bound: 0.0016423
time: 1.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0058647, 0.0059036
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016535, 0.0016645
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0121998, 0.0122808
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016144, 0.0016252
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0091779, 0.0091174
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025499, 0.0025331
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023145, 0.0022993
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0086374, 0.0085805
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0066782, 0.0067225
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005800, 0.0005762

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016485, upper bound: 0.0016286
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016472, upper bound: 0.0016308
time: 1.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0059036, 0.0058647
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016645, 0.0016535
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0122808, 0.0121998
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016252, 0.0016144
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0091174, 0.0091779
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025331, 0.0025499
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0022993, 0.0023145
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0085805, 0.0086374
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0067225, 0.0066782
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005762, 0.0005800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016309, upper bound: 0.0016473
time: 2.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016284, upper bound: 0.0016486
time: 1.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0058876, 0.0058809
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016599, 0.0016580
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0122474, 0.0122334
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016208, 0.0016189
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0091425, 0.0091530
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025400, 0.0025430
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023056, 0.0023082
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0086041, 0.0086140
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0067043, 0.0066966
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005777, 0.0005784

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016423, upper bound: 0.0016373
time: 1.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016409, upper bound: 0.0016404
time: 2.23 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 5.46 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.46
Output dim: 5, lower bound: -0.0016407, upper bound: 0.0016411
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.46
Output dim: 5, lower bound: -0.0016370, upper bound: 0.0016423
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.46
Output dim: 5, lower bound: -0.0016485, upper bound: 0.0016286
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.46
Output dim: 5, lower bound: -0.0016472, upper bound: 0.0016308
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.46
Output dim: 5, lower bound: -0.0016309, upper bound: 0.0016473
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.46
Output dim: 5, lower bound: -0.0016284, upper bound: 0.0016486
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.46
Output dim: 5, lower bound: -0.0016423, upper bound: 0.0016373
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.46
Output dim: 5, lower bound: -0.0016409, upper bound: 0.0016404

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0058157, 0.0058431
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016397, 0.0016474
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0120979, 0.0121548
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016010, 0.0016085
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0090837, 0.0090412
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025237, 0.0025119
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0022908, 0.0022801
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0085488, 0.0085088
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0066224, 0.0066535
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005740, 0.0005713

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015950, upper bound: 0.0015968
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015958, upper bound: 0.0015956
time: 1.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0058327, 0.0058225
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016444, 0.0016416
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0121332, 0.0121120
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016056, 0.0016028
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0090517, 0.0090676
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025148, 0.0025192
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0022827, 0.0022867
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0085187, 0.0085336
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0066417, 0.0066301
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005720, 0.0005730

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015917, upper bound: 0.0015972
time: 1.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015935, upper bound: 0.0015970
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0057996, 0.0058567
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016351, 0.0016512
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0120643, 0.0121832
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0015965, 0.0016123
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0091050, 0.0090161
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025296, 0.0025049
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0022961, 0.0022737
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0085688, 0.0084852
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0066040, 0.0066691
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005754, 0.0005698

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016031, upper bound: 0.0015844
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016042, upper bound: 0.0015833
time: 1.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0058187, 0.0058385
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016405, 0.0016461
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0121042, 0.0121453
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016018, 0.0016072
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0090766, 0.0090459
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025218, 0.0025132
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0022890, 0.0022812
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0085421, 0.0085132
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0066258, 0.0066484
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005736, 0.0005716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016019, upper bound: 0.0015857
time: 1.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016034, upper bound: 0.0015850
time: 1.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0058385, 0.0058187
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016461, 0.0016405
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0121453, 0.0121042
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016072, 0.0016018
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0090459, 0.0090766
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025132, 0.0025218
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0022812, 0.0022890
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0085132, 0.0085421
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0066484, 0.0066258
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005716, 0.0005736

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015855, upper bound: 0.0016032
time: 1.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015859, upper bound: 0.0016016
time: 1.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0058567, 0.0057996
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016512, 0.0016351
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0121832, 0.0120643
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016123, 0.0015965
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0090161, 0.0091050
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025049, 0.0025296
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0022737, 0.0022961
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0084852, 0.0085688
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0066691, 0.0066040
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005698, 0.0005754

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015831, upper bound: 0.0016042
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015844, upper bound: 0.0016029
time: 1.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0058225, 0.0058327
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016416, 0.0016444
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0121120, 0.0121332
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016028, 0.0016056
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0090676, 0.0090517
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025192, 0.0025148
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0022867, 0.0022827
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0085336, 0.0085187
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0066301, 0.0066417
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005730, 0.0005720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015967, upper bound: 0.0015936
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015975, upper bound: 0.0015919
time: 1.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0058431, 0.0058157
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016474, 0.0016397
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0121548, 0.0120979
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016085, 0.0016010
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0090412, 0.0090837
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025119, 0.0025237
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0022801, 0.0022908
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0085088, 0.0085488
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0066535, 0.0066224
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005713, 0.0005740

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015956, upper bound: 0.0015958
time: 1.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015968, upper bound: 0.0015950
time: 1.32 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.44 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 5, lower bound: -0.0015950, upper bound: 0.0015968
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 5, lower bound: -0.0015958, upper bound: 0.0015956
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 5, lower bound: -0.0015917, upper bound: 0.0015972
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 5, lower bound: -0.0015935, upper bound: 0.0015970
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 5, lower bound: -0.0016031, upper bound: 0.0015844
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 5, lower bound: -0.0016042, upper bound: 0.0015833
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 5, lower bound: -0.0016019, upper bound: 0.0015857
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 5, lower bound: -0.0016034, upper bound: 0.0015850
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 5, lower bound: -0.0015855, upper bound: 0.0016032
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 5, lower bound: -0.0015859, upper bound: 0.0016016
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 5, lower bound: -0.0015831, upper bound: 0.0016042
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 5, lower bound: -0.0015844, upper bound: 0.0016029
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 5, lower bound: -0.0015967, upper bound: 0.0015936
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 5, lower bound: -0.0015975, upper bound: 0.0015919
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 5, lower bound: -0.0015956, upper bound: 0.0015958
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 5, lower bound: -0.0015968, upper bound: 0.0015950

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0055790, 0.0056122
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0015729, 0.0015823
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0116054, 0.0116745
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0015358, 0.0015449
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0087248, 0.0086731
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0024240, 0.0024097
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0022003, 0.0021872
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0082110, 0.0081624
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0063528, 0.0063907
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005514, 0.0005481

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014694, upper bound: 0.0014816
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014694, upper bound: 0.0014816
time: 1.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0055849, 0.0055961
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0015746, 0.0015778
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0116177, 0.0116411
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0015374, 0.0015405
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0086998, 0.0086823
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0024171, 0.0024122
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0021940, 0.0021896
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0081875, 0.0081710
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0063595, 0.0063724
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005498, 0.0005487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014710, upper bound: 0.0014796
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014710, upper bound: 0.0014796
time: 1.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0055893, 0.0055916
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0015758, 0.0015765
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0116269, 0.0116317
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0015386, 0.0015393
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0086928, 0.0086892
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0024151, 0.0024141
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0021922, 0.0021913
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0081809, 0.0081775
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0063646, 0.0063672
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005493, 0.0005491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014688, upper bound: 0.0014816
time: 1.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014688, upper bound: 0.0014816
time: 1.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0056018, 0.0055798
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0015794, 0.0015732
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0116529, 0.0116071
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0015421, 0.0015360
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0086744, 0.0087087
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0024100, 0.0024195
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0021876, 0.0021962
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0081636, 0.0081958
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0063788, 0.0063538
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005482, 0.0005503

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014707, upper bound: 0.0014802
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014707, upper bound: 0.0014802
time: 1.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0055568, 0.0056259
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0015667, 0.0015861
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0115593, 0.0117030
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0015297, 0.0015487
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0087461, 0.0086387
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0024299, 0.0024001
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0022056, 0.0021786
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0082310, 0.0081300
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0063276, 0.0064062
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005527, 0.0005459

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014812, upper bound: 0.0014690
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014812, upper bound: 0.0014690
time: 1.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0055687, 0.0056131
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0015700, 0.0015825
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0115841, 0.0116763
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0015330, 0.0015452
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0087261, 0.0086572
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0024244, 0.0024052
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0022006, 0.0021832
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0082123, 0.0081474
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0063411, 0.0063916
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005514, 0.0005471

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014830, upper bound: 0.0014675
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014830, upper bound: 0.0014675
time: 1.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0055727, 0.0056077
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0015712, 0.0015810
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0115924, 0.0116651
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0015341, 0.0015437
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0087177, 0.0086634
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0024220, 0.0024070
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0021985, 0.0021848
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0082044, 0.0081533
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0063457, 0.0063855
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005509, 0.0005475

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014809, upper bound: 0.0014692
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014809, upper bound: 0.0014690
time: 1.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0055879, 0.0056009
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0015754, 0.0015791
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0116239, 0.0116511
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0015382, 0.0015418
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0087073, 0.0086870
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0024191, 0.0024135
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0021958, 0.0021907
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0081945, 0.0081754
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0063630, 0.0063778
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005502, 0.0005490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014830, upper bound: 0.0014679
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014830, upper bound: 0.0014679
time: 1.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0056009, 0.0055879
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0015791, 0.0015754
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0116511, 0.0116239
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0015418, 0.0015382
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0086870, 0.0087073
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0024135, 0.0024191
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0021907, 0.0021958
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0081754, 0.0081945
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0063778, 0.0063630
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005490, 0.0005502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014679, upper bound: 0.0014830
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014679, upper bound: 0.0014830
time: 1.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0056077, 0.0055727
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0015810, 0.0015712
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0116651, 0.0115924
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0015437, 0.0015341
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0086634, 0.0087177
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0024070, 0.0024220
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0021848, 0.0021985
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0081533, 0.0082044
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0063855, 0.0063457
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005475, 0.0005509

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014693, upper bound: 0.0014807
time: 1.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014693, upper bound: 0.0014807
time: 1.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0056131, 0.0055687
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0015825, 0.0015700
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0116763, 0.0115841
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0015452, 0.0015330
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0086572, 0.0087261
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0024052, 0.0024244
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0021832, 0.0022006
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0081474, 0.0082123
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0063916, 0.0063411
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005471, 0.0005514

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014673, upper bound: 0.0014827
time: 1.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014673, upper bound: 0.0014827
time: 1.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0056259, 0.0055568
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0015861, 0.0015667
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0117030, 0.0115593
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0015487, 0.0015297
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0086387, 0.0087461
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0024001, 0.0024299
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0021786, 0.0022056
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0081300, 0.0082310
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0064062, 0.0063276
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005459, 0.0005527

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014690, upper bound: 0.0014814
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014690, upper bound: 0.0014814
time: 1.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0055798, 0.0056018
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0015732, 0.0015794
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0116071, 0.0116529
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0015360, 0.0015421
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0087087, 0.0086744
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0024195, 0.0024100
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0021962, 0.0021876
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0081958, 0.0081636
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0063538, 0.0063788
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005503, 0.0005482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014802, upper bound: 0.0014704
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014802, upper bound: 0.0014704
time: 1.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0055916, 0.0055893
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0015765, 0.0015758
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0116317, 0.0116269
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0015393, 0.0015386
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0086892, 0.0086928
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0024141, 0.0024151
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0021913, 0.0021922
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0081775, 0.0081809
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0063672, 0.0063646
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005491, 0.0005493

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014816, upper bound: 0.0014685
time: 1.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014816, upper bound: 0.0014685
time: 1.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0055961, 0.0055849
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0015778, 0.0015746
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0116411, 0.0116177
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0015405, 0.0015374
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0086823, 0.0086998
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0024122, 0.0024171
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0021896, 0.0021940
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0081710, 0.0081875
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0063724, 0.0063595
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005487, 0.0005498

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014798, upper bound: 0.0014710
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014798, upper bound: 0.0014710
time: 1.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0056122, 0.0055790
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0015823, 0.0015729
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0116745, 0.0116054
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0015449, 0.0015358
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0086731, 0.0087248
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0024097, 0.0024240
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0021872, 0.0022003
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0081624, 0.0082110
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0063907, 0.0063528
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005481, 0.0005514

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014816, upper bound: 0.0014694
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014816, upper bound: 0.0014694
time: 1.61 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.69 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 5, lower bound: -0.0014694, upper bound: 0.0014816
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 5, lower bound: -0.0014694, upper bound: 0.0014816
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 5, lower bound: -0.0014710, upper bound: 0.0014796
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 5, lower bound: -0.0014710, upper bound: 0.0014796
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 5, lower bound: -0.0014688, upper bound: 0.0014816
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 5, lower bound: -0.0014688, upper bound: 0.0014816
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 5, lower bound: -0.0014707, upper bound: 0.0014802
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 5, lower bound: -0.0014707, upper bound: 0.0014802
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 5, lower bound: -0.0014812, upper bound: 0.0014690
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 5, lower bound: -0.0014812, upper bound: 0.0014690
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 5, lower bound: -0.0014830, upper bound: 0.0014675
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 5, lower bound: -0.0014830, upper bound: 0.0014675
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 5, lower bound: -0.0014809, upper bound: 0.0014692
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 5, lower bound: -0.0014809, upper bound: 0.0014690
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 5, lower bound: -0.0014830, upper bound: 0.0014679
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 5, lower bound: -0.0014830, upper bound: 0.0014679
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 5, lower bound: -0.0014679, upper bound: 0.0014830
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 5, lower bound: -0.0014679, upper bound: 0.0014830
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 5, lower bound: -0.0014693, upper bound: 0.0014807
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 5, lower bound: -0.0014693, upper bound: 0.0014807
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 5, lower bound: -0.0014673, upper bound: 0.0014827
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 5, lower bound: -0.0014673, upper bound: 0.0014827
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 5, lower bound: -0.0014690, upper bound: 0.0014814
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 5, lower bound: -0.0014690, upper bound: 0.0014814
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 5, lower bound: -0.0014802, upper bound: 0.0014704
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 5, lower bound: -0.0014802, upper bound: 0.0014704
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 5, lower bound: -0.0014816, upper bound: 0.0014685
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 5, lower bound: -0.0014816, upper bound: 0.0014685
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 5, lower bound: -0.0014798, upper bound: 0.0014710
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 5, lower bound: -0.0014798, upper bound: 0.0014710
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 5, lower bound: -0.0014816, upper bound: 0.0014694
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 5, lower bound: -0.0014816, upper bound: 0.0014694

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 4.26 + 144.69 = 148.95 seconds
