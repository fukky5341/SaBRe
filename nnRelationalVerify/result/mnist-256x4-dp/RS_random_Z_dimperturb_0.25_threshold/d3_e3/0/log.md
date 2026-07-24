## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00079709


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007999, 0.0007999)
1: (-0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0020299, 0.0020299)
2: (0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0012594, 0.0012594)
3: (0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0023515, 0.0023515)
4: (-0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0020648, 0.0020648)
5: (0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007821, 0.0007821)
6: (0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0029844, 0.0029844)
7: (0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0020883, 0.0020883)
8: (-0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0022390, 0.0022390)
9: (-0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0014790, 0.0014790)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.53 + 1.51 = 3.04 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0012113, upper bound: 0.0012113

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0012025, upper bound: 0.0011373
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011373, upper bound: 0.0012025
time: 0.62 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.27 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.27
Output dim: 7, lower bound: -0.0012025, upper bound: 0.0011373
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.27
Output dim: 7, lower bound: -0.0011373, upper bound: 0.0012025

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007875, 0.0007954
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0019983, 0.0020186
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0012398, 0.0012523
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0023384, 0.0023150
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0020326, 0.0020532
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007699, 0.0007777
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0029677, 0.0029380
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0020767, 0.0020559
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0022265, 0.0022042
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0014560, 0.0014707

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011635, upper bound: 0.0010967
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011634, upper bound: 0.0010963
time: 0.63 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007954, 0.0007875
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0020186, 0.0019983
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0012523, 0.0012398
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0023150, 0.0023384
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0020532, 0.0020326
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007777, 0.0007699
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0029380, 0.0029677
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0020559, 0.0020767
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0022042, 0.0022265
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0014707, 0.0014560

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010764, upper bound: 0.0011429
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010768, upper bound: 0.0011414
time: 0.62 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.71 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.71
Output dim: 7, lower bound: -0.0011635, upper bound: 0.0010967
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.71
Output dim: 7, lower bound: -0.0011634, upper bound: 0.0010963
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.71
Output dim: 7, lower bound: -0.0010764, upper bound: 0.0011429
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.71
Output dim: 7, lower bound: -0.0010768, upper bound: 0.0011414

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007383, 0.0007501
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018737, 0.0019034
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011624, 0.0011809
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0022050, 0.0021705
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0019058, 0.0019361
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007219, 0.0007333
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027985, 0.0027547
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019582, 0.0019276
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020995, 0.0020667
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013652, 0.0013869

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011413, upper bound: 0.0010653
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011208, upper bound: 0.0010746
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007421, 0.0007475
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018832, 0.0018968
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011683, 0.0011768
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021973, 0.0021816
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0019155, 0.0019293
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007255, 0.0007308
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027887, 0.0027687
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019514, 0.0019374
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020922, 0.0020772
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013721, 0.0013820

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011044, upper bound: 0.0010591
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011272, upper bound: 0.0010454
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007874, 0.0007752
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0019981, 0.0019671
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0012396, 0.0012204
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0022788, 0.0023147
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0020324, 0.0020009
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007698, 0.0007579
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0028921, 0.0029376
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0020237, 0.0020556
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0021698, 0.0022039
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0014558, 0.0014333

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010336, upper bound: 0.0011032
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010337, upper bound: 0.0011031
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007831, 0.0007875
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0019873, 0.0019983
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0012329, 0.0012398
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0023150, 0.0023022
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0020214, 0.0020326
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007657, 0.0007699
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0029380, 0.0029218
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0020559, 0.0020445
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0022042, 0.0021921
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0014480, 0.0014560

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010630, upper bound: 0.0011264
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010604, upper bound: 0.0011286
time: 0.62 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.65 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 7, lower bound: -0.0011413, upper bound: 0.0010653
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 7, lower bound: -0.0011208, upper bound: 0.0010746
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 7, lower bound: -0.0011044, upper bound: 0.0010591
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 7, lower bound: -0.0011272, upper bound: 0.0010454
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 7, lower bound: -0.0010336, upper bound: 0.0011032
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 7, lower bound: -0.0010337, upper bound: 0.0011031
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 7, lower bound: -0.0010630, upper bound: 0.0011264
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 7, lower bound: -0.0010604, upper bound: 0.0011286

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007224, 0.0007369
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018331, 0.0018699
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011373, 0.0011601
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021662, 0.0021236
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018646, 0.0019020
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007063, 0.0007204
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027492, 0.0026951
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019238, 0.0018859
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020626, 0.0020220
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013356, 0.0013625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0006374, upper bound: 0.0005967
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0006374, upper bound: 0.0005967
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007251, 0.0007337
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018402, 0.0018619
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011416, 0.0011551
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021569, 0.0021317
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018718, 0.0018939
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007090, 0.0007173
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027374, 0.0027055
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019155, 0.0018931
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020537, 0.0020298
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013408, 0.0013566

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011089, upper bound: 0.0010627
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011086, upper bound: 0.0010625
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007386, 0.0007319
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018742, 0.0018573
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011628, 0.0011523
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021516, 0.0021712
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0019064, 0.0018892
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007221, 0.0007156
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027307, 0.0027555
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019108, 0.0019282
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020487, 0.0020673
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013656, 0.0013533

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010895, upper bound: 0.0010391
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010842, upper bound: 0.0010443
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007266, 0.0007426
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018437, 0.0018843
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011439, 0.0011691
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021829, 0.0021359
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018754, 0.0019167
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007103, 0.0007260
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027704, 0.0027107
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019386, 0.0018968
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020785, 0.0020337
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013434, 0.0013730

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010871, upper bound: 0.0010072
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010644, upper bound: 0.0010128
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007390, 0.0007297
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018754, 0.0018517
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011635, 0.0011488
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021451, 0.0021725
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0019076, 0.0018835
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007225, 0.0007134
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027224, 0.0027572
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019050, 0.0019294
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020425, 0.0020686
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013664, 0.0013492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009657, upper bound: 0.0010155
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009653, upper bound: 0.0010165
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007419, 0.0007259
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018827, 0.0018422
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011680, 0.0011429
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021341, 0.0021810
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0019150, 0.0018738
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007254, 0.0007097
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027084, 0.0027680
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018952, 0.0019369
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020320, 0.0020767
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013717, 0.0013422

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010230, upper bound: 0.0010807
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010221, upper bound: 0.0010923
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007788, 0.0007841
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0019763, 0.0019899
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0012261, 0.0012345
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0023051, 0.0022894
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0020102, 0.0020240
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007614, 0.0007666
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0029255, 0.0029056
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0020471, 0.0020332
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0021949, 0.0021799
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0014400, 0.0014498

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009953, upper bound: 0.0010408
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009947, upper bound: 0.0010416
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007797, 0.0007832
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0019786, 0.0019876
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0012275, 0.0012331
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0023025, 0.0022921
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0020126, 0.0020217
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007623, 0.0007658
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0029222, 0.0029090
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0020448, 0.0020356
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0021924, 0.0021825
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0014416, 0.0014482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010443, upper bound: 0.0011093
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010424, upper bound: 0.0011122
time: 0.67 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.61 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.61
Output dim: 7, lower bound: -0.0006374, upper bound: 0.0005967
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.61
Output dim: 7, lower bound: -0.0006374, upper bound: 0.0005967
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.61
Output dim: 7, lower bound: -0.0011089, upper bound: 0.0010627
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.61
Output dim: 7, lower bound: -0.0011086, upper bound: 0.0010625
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.61
Output dim: 7, lower bound: -0.0010895, upper bound: 0.0010391
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.61
Output dim: 7, lower bound: -0.0010842, upper bound: 0.0010443
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.61
Output dim: 7, lower bound: -0.0010871, upper bound: 0.0010072
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.61
Output dim: 7, lower bound: -0.0010644, upper bound: 0.0010128
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.61
Output dim: 7, lower bound: -0.0009657, upper bound: 0.0010155
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.61
Output dim: 7, lower bound: -0.0009653, upper bound: 0.0010165
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.61
Output dim: 7, lower bound: -0.0010230, upper bound: 0.0010807
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.61
Output dim: 7, lower bound: -0.0010221, upper bound: 0.0010923
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.61
Output dim: 7, lower bound: -0.0009953, upper bound: 0.0010408
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.61
Output dim: 7, lower bound: -0.0009947, upper bound: 0.0010416
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.61
Output dim: 7, lower bound: -0.0010443, upper bound: 0.0011093
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.61
Output dim: 7, lower bound: -0.0010424, upper bound: 0.0011122

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007223, 0.0007310
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018330, 0.0018550
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011372, 0.0011508
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021489, 0.0021234
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018644, 0.0018868
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007062, 0.0007147
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027272, 0.0026949
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019084, 0.0018857
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020461, 0.0020218
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013355, 0.0013516

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010925, upper bound: 0.0010444
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010920, upper bound: 0.0010481
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007224, 0.0007301
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018332, 0.0018527
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011373, 0.0011494
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021463, 0.0021237
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018647, 0.0018845
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007063, 0.0007138
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027239, 0.0026953
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019061, 0.0018860
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020436, 0.0020221
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013357, 0.0013499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010736, upper bound: 0.0010243
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010652, upper bound: 0.0010308
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007343, 0.0007280
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018635, 0.0018474
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011561, 0.0011461
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021401, 0.0021588
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018955, 0.0018791
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007180, 0.0007117
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027160, 0.0027398
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019005, 0.0019172
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020377, 0.0020555
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013578, 0.0013460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010671, upper bound: 0.0010017
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010525, upper bound: 0.0010166
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007345, 0.0007277
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018638, 0.0018466
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011563, 0.0011456
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021392, 0.0021591
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018958, 0.0018783
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007181, 0.0007114
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027149, 0.0027402
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018998, 0.0019175
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020368, 0.0020558
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013580, 0.0013454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010630, upper bound: 0.0010038
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010502, upper bound: 0.0010210
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007219, 0.0007400
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018319, 0.0018779
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011365, 0.0011651
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021755, 0.0021222
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018634, 0.0019101
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007058, 0.0007235
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027609, 0.0026933
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019320, 0.0018846
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020714, 0.0020206
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013347, 0.0013683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010778, upper bound: 0.0009952
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010658, upper bound: 0.0009976
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007240, 0.0007426
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018373, 0.0018843
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011399, 0.0011691
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021829, 0.0021284
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018688, 0.0019167
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007079, 0.0007260
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027704, 0.0027012
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019386, 0.0018902
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020785, 0.0020266
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013387, 0.0013730

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010479, upper bound: 0.0009942
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010476, upper bound: 0.0009973
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007346, 0.0007272
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018643, 0.0018453
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011566, 0.0011448
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021377, 0.0021597
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018963, 0.0018770
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007183, 0.0007109
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027130, 0.0027409
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018984, 0.0019180
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020354, 0.0020563
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013583, 0.0013445

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009170, upper bound: 0.0009741
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009224, upper bound: 0.0009653
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007365, 0.0007297
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018690, 0.0018517
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011595, 0.0011488
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021451, 0.0021651
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0019011, 0.0018835
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007201, 0.0007134
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027224, 0.0027478
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019050, 0.0019228
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020425, 0.0020615
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013618, 0.0013492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009437, upper bound: 0.0009836
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009357, upper bound: 0.0009946
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007400, 0.0007246
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018779, 0.0018387
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011651, 0.0011407
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021300, 0.0021754
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0019101, 0.0018702
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007235, 0.0007084
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027033, 0.0027609
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018916, 0.0019320
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020281, 0.0020714
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013683, 0.0013397

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009741, upper bound: 0.0010461
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009852, upper bound: 0.0010178
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007405, 0.0007241
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018792, 0.0018375
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011659, 0.0011400
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021286, 0.0021769
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0019114, 0.0018690
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007240, 0.0007079
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027015, 0.0027628
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018904, 0.0019333
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020268, 0.0020728
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013692, 0.0013388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009719, upper bound: 0.0010557
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009846, upper bound: 0.0010336
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007744, 0.0007818
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0019652, 0.0019838
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0012192, 0.0012308
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0022981, 0.0022766
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0019990, 0.0020179
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007572, 0.0007643
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0029166, 0.0028893
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0020409, 0.0020218
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0021882, 0.0021677
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0014319, 0.0014454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009851, upper bound: 0.0010235
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009838, upper bound: 0.0010309
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007764, 0.0007841
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0019701, 0.0019899
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0012223, 0.0012345
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0023051, 0.0022823
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0020039, 0.0020240
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007590, 0.0007666
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0029255, 0.0028965
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0020471, 0.0020268
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0021949, 0.0021731
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0014355, 0.0014498

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009468, upper bound: 0.0010010
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009510, upper bound: 0.0009908
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007760, 0.0007795
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0019691, 0.0019781
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0012216, 0.0012272
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0022916, 0.0022811
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0020029, 0.0020121
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007586, 0.0007621
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0029083, 0.0028950
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0020351, 0.0020258
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0021819, 0.0021720
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0014347, 0.0014413

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009790, upper bound: 0.0010247
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009781, upper bound: 0.0010253
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007761, 0.0007793
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0019695, 0.0019776
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0012219, 0.0012269
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0022910, 0.0022815
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0020033, 0.0020116
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007588, 0.0007619
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0029076, 0.0028956
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0020346, 0.0020262
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0021814, 0.0021724
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0014350, 0.0014409

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010204, upper bound: 0.0010677
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010097, upper bound: 0.0010898
time: 0.64 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.73 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 7, lower bound: -0.0010925, upper bound: 0.0010444
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 7, lower bound: -0.0010920, upper bound: 0.0010481
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 7, lower bound: -0.0010736, upper bound: 0.0010243
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 7, lower bound: -0.0010652, upper bound: 0.0010308
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 7, lower bound: -0.0010671, upper bound: 0.0010017
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 7, lower bound: -0.0010525, upper bound: 0.0010166
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 7, lower bound: -0.0010630, upper bound: 0.0010038
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 7, lower bound: -0.0010502, upper bound: 0.0010210
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 7, lower bound: -0.0010778, upper bound: 0.0009952
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 7, lower bound: -0.0010658, upper bound: 0.0009976
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 7, lower bound: -0.0010479, upper bound: 0.0009942
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 7, lower bound: -0.0010476, upper bound: 0.0009973
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 7, lower bound: -0.0009170, upper bound: 0.0009741
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 7, lower bound: -0.0009224, upper bound: 0.0009653
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 7, lower bound: -0.0009437, upper bound: 0.0009836
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 7, lower bound: -0.0009357, upper bound: 0.0009946
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 7, lower bound: -0.0009741, upper bound: 0.0010461
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 7, lower bound: -0.0009852, upper bound: 0.0010178
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 7, lower bound: -0.0009719, upper bound: 0.0010557
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 7, lower bound: -0.0009846, upper bound: 0.0010336
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 7, lower bound: -0.0009851, upper bound: 0.0010235
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 7, lower bound: -0.0009838, upper bound: 0.0010309
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 7, lower bound: -0.0009468, upper bound: 0.0010010
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 7, lower bound: -0.0009510, upper bound: 0.0009908
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 7, lower bound: -0.0009790, upper bound: 0.0010247
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 7, lower bound: -0.0009781, upper bound: 0.0010253
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 7, lower bound: -0.0010204, upper bound: 0.0010677
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 7, lower bound: -0.0010097, upper bound: 0.0010898

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007178, 0.0007266
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018215, 0.0018438
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011301, 0.0011439
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021359, 0.0021101
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018528, 0.0018754
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007018, 0.0007104
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027108, 0.0026780
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018969, 0.0018740
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020337, 0.0020092
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013272, 0.0013434

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010812, upper bound: 0.0010327
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010675, upper bound: 0.0010337
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007180, 0.0007265
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018221, 0.0018435
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011305, 0.0011437
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021356, 0.0021109
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018534, 0.0018752
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007020, 0.0007103
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027104, 0.0026790
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018966, 0.0018746
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020335, 0.0020099
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013276, 0.0013432

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010565, upper bound: 0.0010081
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010481, upper bound: 0.0010153
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007184, 0.0007278
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018231, 0.0018470
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011310, 0.0011459
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021397, 0.0021119
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018543, 0.0018787
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007024, 0.0007116
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027155, 0.0026803
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019002, 0.0018755
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020373, 0.0020109
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013283, 0.0013458

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010643, upper bound: 0.0010128
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010493, upper bound: 0.0010148
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007202, 0.0007301
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018275, 0.0018527
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011338, 0.0011494
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021463, 0.0021171
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018589, 0.0018845
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007041, 0.0007138
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027239, 0.0026868
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019061, 0.0018801
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020436, 0.0020158
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013315, 0.0013499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010134, upper bound: 0.0009892
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010236, upper bound: 0.0009829
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007129, 0.0007106
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018090, 0.0018032
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011223, 0.0011187
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020889, 0.0020956
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018401, 0.0018342
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006970, 0.0006947
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026511, 0.0026596
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018551, 0.0018611
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019890, 0.0019954
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013181, 0.0013138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010547, upper bound: 0.0009896
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010526, upper bound: 0.0009896
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007169, 0.0007076
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018194, 0.0017957
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011287, 0.0011141
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020802, 0.0021076
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018506, 0.0018265
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007010, 0.0006918
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026401, 0.0026749
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018474, 0.0018717
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019807, 0.0020068
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013256, 0.0013084

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010416, upper bound: 0.0010061
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010240, upper bound: 0.0010066
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007129, 0.0007103
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018091, 0.0018024
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011224, 0.0011182
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020880, 0.0020957
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018401, 0.0018334
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006970, 0.0006944
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026500, 0.0026597
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018543, 0.0018612
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019881, 0.0019955
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013181, 0.0013133

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010289, upper bound: 0.0009614
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010209, upper bound: 0.0009698
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007171, 0.0007074
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018197, 0.0017951
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011289, 0.0011137
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020796, 0.0021080
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018509, 0.0018259
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007011, 0.0006916
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026392, 0.0026753
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018468, 0.0018720
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019801, 0.0020071
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013258, 0.0013079

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0005661, upper bound: 0.0005358
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0005661, upper bound: 0.0005358
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007203, 0.0007388
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018278, 0.0018748
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011340, 0.0011631
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021719, 0.0021175
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018592, 0.0019070
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007042, 0.0007223
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027564, 0.0026873
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019288, 0.0018805
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020680, 0.0020161
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013318, 0.0013660

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010556, upper bound: 0.0009649
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010368, upper bound: 0.0009738
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007207, 0.0007383
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018288, 0.0018736
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011346, 0.0011624
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021704, 0.0021186
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018602, 0.0019057
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007046, 0.0007218
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027546, 0.0026888
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019275, 0.0018815
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020666, 0.0020172
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013325, 0.0013651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010499, upper bound: 0.0009804
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010461, upper bound: 0.0009811
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007198, 0.0007386
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018266, 0.0018744
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011332, 0.0011629
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021714, 0.0021160
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018579, 0.0019066
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007037, 0.0007222
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027558, 0.0026854
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019284, 0.0018791
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020675, 0.0020147
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013309, 0.0013657

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009571, upper bound: 0.0009001
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009571, upper bound: 0.0009001
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007200, 0.0007383
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018271, 0.0018736
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011335, 0.0011624
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021705, 0.0021166
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018584, 0.0019058
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007039, 0.0007219
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027546, 0.0026862
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019276, 0.0018797
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020666, 0.0020153
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013312, 0.0013651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010382, upper bound: 0.0009856
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010264, upper bound: 0.0009877
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007290, 0.0007111
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018499, 0.0018045
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011477, 0.0011195
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020904, 0.0021431
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018817, 0.0018355
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007127, 0.0006952
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026530, 0.0027198
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018565, 0.0019032
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019904, 0.0020405
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013479, 0.0013148

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008956, upper bound: 0.0009400
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008888, upper bound: 0.0009521
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007186, 0.0007231
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018235, 0.0018350
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011313, 0.0011384
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021258, 0.0021124
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018548, 0.0018665
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007025, 0.0007070
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026979, 0.0026809
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018878, 0.0018760
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020241, 0.0020114
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013286, 0.0013370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009009, upper bound: 0.0009329
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008904, upper bound: 0.0009439
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007207, 0.0007161
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018288, 0.0018172
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011346, 0.0011274
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021051, 0.0021186
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018602, 0.0018484
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007046, 0.0007001
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026716, 0.0026888
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018695, 0.0018815
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020044, 0.0020172
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013325, 0.0013240

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009312, upper bound: 0.0009710
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009312, upper bound: 0.0009704
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007230, 0.0007125
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018347, 0.0018080
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011383, 0.0011217
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020945, 0.0021254
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018662, 0.0018390
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007069, 0.0006966
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026581, 0.0026974
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018600, 0.0018875
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019942, 0.0020237
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013368, 0.0013173

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008888, upper bound: 0.0009527
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008904, upper bound: 0.0009439
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007373, 0.0007087
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018709, 0.0017983
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011607, 0.0011157
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020833, 0.0021673
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0019030, 0.0018292
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007208, 0.0006928
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026439, 0.0027506
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018501, 0.0019247
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019836, 0.0020636
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013631, 0.0013103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009618, upper bound: 0.0010322
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009618, upper bound: 0.0010335
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007241, 0.0007199
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018375, 0.0018269
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011400, 0.0011334
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021164, 0.0021287
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018691, 0.0018583
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007080, 0.0007039
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026860, 0.0027016
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018795, 0.0018904
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020151, 0.0020269
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013389, 0.0013311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009126, upper bound: 0.0009434
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009122, upper bound: 0.0009439
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007377, 0.0007082
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018719, 0.0017971
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011614, 0.0011149
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020819, 0.0021686
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0019041, 0.0018280
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007212, 0.0006924
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026422, 0.0027522
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018489, 0.0019258
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019823, 0.0020648
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013639, 0.0013094

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009567, upper bound: 0.0010357
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009521, upper bound: 0.0010400
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007246, 0.0007195
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018388, 0.0018258
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011408, 0.0011327
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021151, 0.0021302
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018704, 0.0018571
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007085, 0.0007034
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026843, 0.0027035
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018783, 0.0018918
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020139, 0.0020283
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013398, 0.0013303

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009120, upper bound: 0.0009523
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009116, upper bound: 0.0009528
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007724, 0.0007802
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0019601, 0.0019799
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0012161, 0.0012283
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0022936, 0.0022707
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0019938, 0.0020139
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007552, 0.0007628
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0029109, 0.0028818
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0020369, 0.0020166
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0021839, 0.0021621
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0014282, 0.0014426

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009432, upper bound: 0.0009861
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009432, upper bound: 0.0009799
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007729, 0.0007799
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0019612, 0.0019790
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0012168, 0.0012278
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0022926, 0.0022720
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0019949, 0.0020130
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007556, 0.0007625
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0029095, 0.0028835
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0020360, 0.0020177
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0021829, 0.0021633
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0014290, 0.0014419

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009678, upper bound: 0.0010140
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009664, upper bound: 0.0010142
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007705, 0.0007668
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0019553, 0.0019458
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0012131, 0.0012072
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0022542, 0.0022652
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0019889, 0.0019792
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007533, 0.0007497
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0028608, 0.0028748
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0020019, 0.0020116
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0021463, 0.0021568
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0014247, 0.0014178

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009254, upper bound: 0.0009672
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009164, upper bound: 0.0009788
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007588, 0.0007788
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0019255, 0.0019763
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011946, 0.0012261
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0022895, 0.0022306
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0019586, 0.0020103
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007419, 0.0007614
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0029057, 0.0028309
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0020332, 0.0019810
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0021800, 0.0021239
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0014030, 0.0014400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009409, upper bound: 0.0009715
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009400, upper bound: 0.0009808
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007715, 0.0007771
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0019579, 0.0019719
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0012147, 0.0012234
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0022844, 0.0022681
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0019915, 0.0020058
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007543, 0.0007597
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0028992, 0.0028785
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0020287, 0.0020143
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0021751, 0.0021596
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0014265, 0.0014368

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009372, upper bound: 0.0009854
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009374, upper bound: 0.0009828
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007735, 0.0007795
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0019629, 0.0019781
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0012178, 0.0012272
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0022916, 0.0022739
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0019966, 0.0020121
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007562, 0.0007621
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0029083, 0.0028859
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0020351, 0.0020194
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0021819, 0.0021651
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0014302, 0.0014413

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009301, upper bound: 0.0009853
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009345, upper bound: 0.0009728
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007593, 0.0007661
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0019267, 0.0019441
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011953, 0.0012061
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0022522, 0.0022320
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0019598, 0.0019775
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007423, 0.0007490
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0028583, 0.0028327
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0020001, 0.0019822
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0021444, 0.0021252
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0014038, 0.0014165

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010097, upper bound: 0.0010441
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010086, upper bound: 0.0010564
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007625, 0.0007625
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0019349, 0.0019350
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0012004, 0.0012005
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0022416, 0.0022415
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0019681, 0.0019682
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007455, 0.0007455
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0028449, 0.0028447
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019907, 0.0019906
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0021344, 0.0021342
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0014098, 0.0014099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009715, upper bound: 0.0010506
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009728, upper bound: 0.0010507
time: 0.65 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.86 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0010812, upper bound: 0.0010327
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0010675, upper bound: 0.0010337
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0010565, upper bound: 0.0010081
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0010481, upper bound: 0.0010153
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0010643, upper bound: 0.0010128
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0010493, upper bound: 0.0010148
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0010134, upper bound: 0.0009892
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0010236, upper bound: 0.0009829
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0010547, upper bound: 0.0009896
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0010526, upper bound: 0.0009896
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0010416, upper bound: 0.0010061
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0010240, upper bound: 0.0010066
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0010289, upper bound: 0.0009614
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0010209, upper bound: 0.0009698
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0005661, upper bound: 0.0005358
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0005661, upper bound: 0.0005358
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0010556, upper bound: 0.0009649
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0010368, upper bound: 0.0009738
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0010499, upper bound: 0.0009804
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0010461, upper bound: 0.0009811
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0009571, upper bound: 0.0009001
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0009571, upper bound: 0.0009001
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0010382, upper bound: 0.0009856
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0010264, upper bound: 0.0009877
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0008956, upper bound: 0.0009400
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0008888, upper bound: 0.0009521
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0009009, upper bound: 0.0009329
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0008904, upper bound: 0.0009439
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0009312, upper bound: 0.0009710
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0009312, upper bound: 0.0009704
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0008888, upper bound: 0.0009527
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0008904, upper bound: 0.0009439
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0009618, upper bound: 0.0010322
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0009618, upper bound: 0.0010335
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0009126, upper bound: 0.0009434
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0009122, upper bound: 0.0009439
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0009567, upper bound: 0.0010357
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0009521, upper bound: 0.0010400
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0009120, upper bound: 0.0009523
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0009116, upper bound: 0.0009528
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0009432, upper bound: 0.0009861
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0009432, upper bound: 0.0009799
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0009678, upper bound: 0.0010140
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0009664, upper bound: 0.0010142
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0009254, upper bound: 0.0009672
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0009164, upper bound: 0.0009788
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0009409, upper bound: 0.0009715
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0009400, upper bound: 0.0009808
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0009372, upper bound: 0.0009854
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0009374, upper bound: 0.0009828
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0009301, upper bound: 0.0009853
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0009345, upper bound: 0.0009728
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0010097, upper bound: 0.0010441
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0010086, upper bound: 0.0010564
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0009715, upper bound: 0.0010506
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 7, lower bound: -0.0009728, upper bound: 0.0010507

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007159, 0.0007251
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018166, 0.0018401
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011270, 0.0011416
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021317, 0.0021045
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018478, 0.0018717
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006999, 0.0007089
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027053, 0.0026709
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018931, 0.0018689
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020297, 0.0020038
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013236, 0.0013407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010275, upper bound: 0.0009938
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010449, upper bound: 0.0009811
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007163, 0.0007247
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018178, 0.0018390
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011278, 0.0011409
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021304, 0.0021059
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018490, 0.0018706
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007004, 0.0007085
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027037, 0.0026726
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018919, 0.0018702
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020285, 0.0020051
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013245, 0.0013399

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010094, upper bound: 0.0009946
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010314, upper bound: 0.0009828
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007139, 0.0007241
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018117, 0.0018376
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011240, 0.0011401
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021288, 0.0020988
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018428, 0.0018691
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006980, 0.0007080
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027017, 0.0026636
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018905, 0.0018639
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020269, 0.0019984
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013200, 0.0013389

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009500, upper bound: 0.0009151
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009500, upper bound: 0.0009151
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007157, 0.0007265
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018162, 0.0018435
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011268, 0.0011437
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021356, 0.0021040
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018474, 0.0018752
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006997, 0.0007103
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027104, 0.0026702
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018966, 0.0018685
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020335, 0.0020033
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013233, 0.0013432

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010385, upper bound: 0.0010050
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010238, upper bound: 0.0010060
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007165, 0.0007264
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018183, 0.0018433
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011281, 0.0011436
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021354, 0.0021064
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018495, 0.0018750
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007006, 0.0007102
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027101, 0.0026733
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018964, 0.0018707
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020332, 0.0020057
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013249, 0.0013431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010478, upper bound: 0.0009959
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010471, upper bound: 0.0009959
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007169, 0.0007260
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018194, 0.0018422
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011287, 0.0011429
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021341, 0.0021076
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018506, 0.0018739
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007010, 0.0007098
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027085, 0.0026749
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018953, 0.0018717
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020320, 0.0020068
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013256, 0.0013423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010333, upper bound: 0.0009978
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010314, upper bound: 0.0009981
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007115, 0.0007104
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018056, 0.0018027
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011202, 0.0011184
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020884, 0.0020916
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018365, 0.0018337
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006956, 0.0006945
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026504, 0.0026546
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018546, 0.0018575
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019885, 0.0019916
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013155, 0.0013135

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009160, upper bound: 0.0008884
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009160, upper bound: 0.0008884
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007002, 0.0007210
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017770, 0.0018296
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011024, 0.0011351
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021195, 0.0020585
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018075, 0.0018610
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006846, 0.0007049
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026899, 0.0026125
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018822, 0.0018281
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020181, 0.0019600
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0012947, 0.0013330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009243, upper bound: 0.0008831
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009243, upper bound: 0.0008831
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007089, 0.0007076
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017989, 0.0017956
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011160, 0.0011140
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020802, 0.0020839
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018298, 0.0018265
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006931, 0.0006918
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026400, 0.0026448
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018474, 0.0018507
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019807, 0.0019842
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013107, 0.0013083

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010222, upper bound: 0.0009519
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010116, upper bound: 0.0009584
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007099, 0.0007074
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018015, 0.0017950
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011176, 0.0011137
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020795, 0.0020869
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018324, 0.0018259
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006941, 0.0006916
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026391, 0.0026485
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018467, 0.0018533
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019800, 0.0019871
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013126, 0.0013079

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009913, upper bound: 0.0009271
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009940, upper bound: 0.0009269
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007152, 0.0007062
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018149, 0.0017920
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011260, 0.0011118
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020760, 0.0021025
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018460, 0.0018228
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006992, 0.0006904
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026347, 0.0026683
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018436, 0.0018671
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019767, 0.0020019
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013224, 0.0013057

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0005558, upper bound: 0.0005191
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0005558, upper bound: 0.0005191
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007155, 0.0007057
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018157, 0.0017907
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011265, 0.0011110
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020744, 0.0021034
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018469, 0.0018214
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006995, 0.0006899
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026327, 0.0026695
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018423, 0.0018680
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019752, 0.0020028
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013229, 0.0013047

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0005428, upper bound: 0.0005255
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0005428, upper bound: 0.0005255
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007083, 0.0007078
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017974, 0.0017961
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011151, 0.0011143
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020807, 0.0020822
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018282, 0.0018270
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006925, 0.0006920
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026407, 0.0026425
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018478, 0.0018491
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019812, 0.0019826
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013096, 0.0013087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009248, upper bound: 0.0008737
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009248, upper bound: 0.0008737
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007104, 0.0007103
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018028, 0.0018024
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011184, 0.0011182
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020880, 0.0020884
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018337, 0.0018334
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006946, 0.0006944
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026500, 0.0026505
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018543, 0.0018547
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019881, 0.0019885
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013135, 0.0013133

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010111, upper bound: 0.0009587
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009928, upper bound: 0.0009601
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0006997, 0.0007217
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017755, 0.0018313
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011015, 0.0011362
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021215, 0.0020569
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018060, 0.0018628
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006841, 0.0007056
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026925, 0.0026104
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018841, 0.0018267
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020200, 0.0019585
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0012937, 0.0013343

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009429, upper bound: 0.0008770
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009429, upper bound: 0.0008770
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007031, 0.0007182
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017843, 0.0018226
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011070, 0.0011307
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021114, 0.0020671
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018150, 0.0018539
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006875, 0.0007022
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026796, 0.0026234
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018751, 0.0018357
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020104, 0.0019682
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013001, 0.0013280

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010208, upper bound: 0.0009567
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010190, upper bound: 0.0009569
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007164, 0.0007344
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018180, 0.0018636
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011279, 0.0011562
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021588, 0.0021060
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018492, 0.0018955
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007004, 0.0007180
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027398, 0.0026728
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019172, 0.0018703
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020555, 0.0020053
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013246, 0.0013578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010273, upper bound: 0.0009500
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010099, upper bound: 0.0009589
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007166, 0.0007340
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018184, 0.0018627
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011281, 0.0011556
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021579, 0.0021065
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018496, 0.0018947
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007006, 0.0007177
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027386, 0.0026734
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019163, 0.0018707
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020546, 0.0020057
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013249, 0.0013572

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010371, upper bound: 0.0009723
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010361, upper bound: 0.0009721
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007111, 0.0007259
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018044, 0.0018420
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011195, 0.0011428
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021339, 0.0020903
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018354, 0.0018736
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006952, 0.0007097
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027082, 0.0026529
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018950, 0.0018564
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020318, 0.0019903
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013147, 0.0013421

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009353, upper bound: 0.0008718
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009231, upper bound: 0.0008787
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007070, 0.0007386
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017942, 0.0018744
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011131, 0.0011629
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021714, 0.0020785
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018250, 0.0019066
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006913, 0.0007222
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027558, 0.0026379
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019284, 0.0018459
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020675, 0.0019790
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013073, 0.0013657

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009353, upper bound: 0.0008718
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009231, upper bound: 0.0008787
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007184, 0.0007371
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018229, 0.0018705
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011310, 0.0011605
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021669, 0.0021118
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018542, 0.0019026
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007023, 0.0007207
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027500, 0.0026801
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019244, 0.0018754
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020632, 0.0020107
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013282, 0.0013629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010169, upper bound: 0.0009553
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010096, upper bound: 0.0009639
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007187, 0.0007366
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018238, 0.0018693
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011315, 0.0011597
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021654, 0.0021128
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018551, 0.0019013
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007027, 0.0007202
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027482, 0.0026815
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019231, 0.0018763
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020618, 0.0020117
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013289, 0.0013620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010174, upper bound: 0.0009783
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010170, upper bound: 0.0009786
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007088, 0.0006934
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017987, 0.0017597
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011159, 0.0010917
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020385, 0.0020837
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018296, 0.0017899
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006930, 0.0006780
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0025871, 0.0026445
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018103, 0.0018505
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019409, 0.0019840
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013106, 0.0012821

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008789, upper bound: 0.0009230
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008787, upper bound: 0.0009231
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007113, 0.0006898
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018051, 0.0017505
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011199, 0.0010860
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020278, 0.0020911
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018361, 0.0017805
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006955, 0.0006744
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0025736, 0.0026539
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018009, 0.0018571
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019308, 0.0019911
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013152, 0.0012754

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008787, upper bound: 0.0009361
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008770, upper bound: 0.0009423
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0006986, 0.0007054
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017728, 0.0017902
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0010998, 0.0011106
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020738, 0.0020537
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018032, 0.0018209
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006830, 0.0006897
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026319, 0.0026064
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018417, 0.0018238
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019746, 0.0019554
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0012917, 0.0013043

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008911, upper bound: 0.0009144
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008904, upper bound: 0.0009227
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007009, 0.0007014
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017786, 0.0017798
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011035, 0.0011042
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020618, 0.0020605
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018092, 0.0018104
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006853, 0.0006857
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026167, 0.0026150
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018311, 0.0018299
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019632, 0.0019619
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0012959, 0.0012968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008779, upper bound: 0.0009308
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008778, upper bound: 0.0009308
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007178, 0.0007134
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018216, 0.0018103
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011301, 0.0011231
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020971, 0.0021102
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018529, 0.0018414
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007018, 0.0006975
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026615, 0.0026781
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018624, 0.0018740
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019968, 0.0020093
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013272, 0.0013190

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009149, upper bound: 0.0009540
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009142, upper bound: 0.0009542
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007180, 0.0007125
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018220, 0.0018080
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011304, 0.0011217
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020945, 0.0021107
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018533, 0.0018390
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007020, 0.0006966
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026582, 0.0026787
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018601, 0.0018744
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019943, 0.0020097
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013275, 0.0013173

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009149, upper bound: 0.0009535
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009142, upper bound: 0.0009540
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007134, 0.0006923
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018103, 0.0017567
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011231, 0.0010899
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020350, 0.0020972
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018414, 0.0017868
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006975, 0.0006768
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0025827, 0.0026616
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018073, 0.0018625
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019377, 0.0019969
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013190, 0.0012799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008720, upper bound: 0.0009358
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008718, upper bound: 0.0009362
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007028, 0.0007038
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017833, 0.0017860
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011064, 0.0011081
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020690, 0.0020659
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018139, 0.0018167
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006871, 0.0006881
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026259, 0.0026219
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018375, 0.0018347
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019700, 0.0019671
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0012994, 0.0013013

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008737, upper bound: 0.0009248
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008735, upper bound: 0.0009281
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007329, 0.0007053
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018598, 0.0017898
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011538, 0.0011104
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020734, 0.0021544
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018917, 0.0018206
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007165, 0.0006896
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026315, 0.0027343
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018414, 0.0019133
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019742, 0.0020514
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013550, 0.0013041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009410, upper bound: 0.0009868
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009323, upper bound: 0.0010097
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007339, 0.0007052
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018624, 0.0017896
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011554, 0.0011103
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020731, 0.0021575
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018944, 0.0018203
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007175, 0.0006895
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026311, 0.0027381
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018411, 0.0019160
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019739, 0.0020543
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013570, 0.0013039

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009461, upper bound: 0.0010111
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009412, upper bound: 0.0010174
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007197, 0.0007174
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018262, 0.0018206
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011330, 0.0011295
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021090, 0.0021156
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018576, 0.0018518
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007036, 0.0007014
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026766, 0.0026850
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018730, 0.0018788
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020081, 0.0020144
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013306, 0.0013265

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008965, upper bound: 0.0009237
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008955, upper bound: 0.0009280
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007216, 0.0007199
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018312, 0.0018269
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011361, 0.0011334
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021164, 0.0021213
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018626, 0.0018583
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007055, 0.0007039
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026860, 0.0026923
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018795, 0.0018839
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020151, 0.0020199
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013342, 0.0013311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008995, upper bound: 0.0009281
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008995, upper bound: 0.0009306
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007336, 0.0007044
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018615, 0.0017874
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011549, 0.0011089
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020707, 0.0021565
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018935, 0.0018181
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007172, 0.0006887
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026279, 0.0027368
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018389, 0.0019151
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019716, 0.0020533
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013563, 0.0013024

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009443, upper bound: 0.0010226
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009445, upper bound: 0.0010230
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007336, 0.0007041
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018616, 0.0017867
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011550, 0.0011085
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020698, 0.0021566
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018936, 0.0018173
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007172, 0.0006884
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026268, 0.0027370
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018381, 0.0019153
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019707, 0.0020535
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013564, 0.0013018

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009396, upper bound: 0.0010268
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009397, upper bound: 0.0010275
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007202, 0.0007170
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018275, 0.0018194
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011338, 0.0011288
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021077, 0.0021171
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018589, 0.0018507
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007041, 0.0007010
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026750, 0.0026868
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018718, 0.0018801
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020069, 0.0020158
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013315, 0.0013256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008992, upper bound: 0.0009379
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008993, upper bound: 0.0009391
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007221, 0.0007195
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018325, 0.0018258
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011369, 0.0011327
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021151, 0.0021228
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018639, 0.0018571
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007060, 0.0007034
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026843, 0.0026942
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018783, 0.0018852
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020139, 0.0020213
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013352, 0.0013303

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008989, upper bound: 0.0009382
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008989, upper bound: 0.0009396
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007254, 0.0007351
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018408, 0.0018653
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011421, 0.0011572
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021609, 0.0021325
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018724, 0.0018973
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007092, 0.0007187
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027424, 0.0027064
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019190, 0.0018938
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020575, 0.0020305
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013412, 0.0013591

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008943, upper bound: 0.0009450
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009000, upper bound: 0.0009341
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007274, 0.0007313
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018459, 0.0018558
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011452, 0.0011513
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021498, 0.0021384
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018776, 0.0018876
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007112, 0.0007150
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027284, 0.0027139
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019092, 0.0018991
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020470, 0.0020361
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013450, 0.0013521

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009270, upper bound: 0.0009629
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009257, upper bound: 0.0009638
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007690, 0.0007760
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0019514, 0.0019691
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0012107, 0.0012216
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0022811, 0.0022606
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0019849, 0.0020029
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007518, 0.0007586
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0028950, 0.0028691
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0020258, 0.0020076
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0021720, 0.0021525
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0014218, 0.0014347

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009461, upper bound: 0.0009807
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009350, upper bound: 0.0009923
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007691, 0.0007758
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0019517, 0.0019686
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0012108, 0.0012213
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0022805, 0.0022609
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0019852, 0.0020024
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007519, 0.0007585
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0028943, 0.0028694
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0020253, 0.0020079
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0021714, 0.0021528
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0014220, 0.0014344

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009249, upper bound: 0.0009759
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009249, upper bound: 0.0009718
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007478, 0.0007487
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018977, 0.0018999
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011774, 0.0011787
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0022010, 0.0021984
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0019303, 0.0019326
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007311, 0.0007320
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027933, 0.0027901
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019546, 0.0019524
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020957, 0.0020933
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013827, 0.0013843

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008831, upper bound: 0.0009285
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008831, upper bound: 0.0009251
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007522, 0.0007451
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0019087, 0.0018907
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011842, 0.0011730
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021903, 0.0022111
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0019415, 0.0019232
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007354, 0.0007285
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027798, 0.0028062
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019452, 0.0019636
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020855, 0.0021053
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013907, 0.0013776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008996, upper bound: 0.0009619
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008994, upper bound: 0.0009619
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007573, 0.0007777
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0019217, 0.0019735
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011923, 0.0012244
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0022862, 0.0022263
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0019547, 0.0020074
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007404, 0.0007604
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0029015, 0.0028254
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0020304, 0.0019771
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0021769, 0.0021197
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0014002, 0.0014379

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008995, upper bound: 0.0009344
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008995, upper bound: 0.0009281
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007577, 0.0007774
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0019228, 0.0019727
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011929, 0.0012239
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0022853, 0.0022275
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0019558, 0.0020066
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007408, 0.0007600
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0029004, 0.0028269
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0020295, 0.0019782
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0021760, 0.0021209
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0014010, 0.0014374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009183, upper bound: 0.0009483
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009067, upper bound: 0.0009591
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007237, 0.0007319
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018366, 0.0018573
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011394, 0.0011523
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021516, 0.0021276
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018681, 0.0018892
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007076, 0.0007156
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027306, 0.0027002
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019108, 0.0018895
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020486, 0.0020258
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013382, 0.0013532

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009154, upper bound: 0.0009525
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009066, upper bound: 0.0009638
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007263, 0.0007290
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018430, 0.0018499
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011434, 0.0011477
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021430, 0.0021351
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018747, 0.0018816
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007101, 0.0007127
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027197, 0.0027097
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019031, 0.0018961
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020405, 0.0020329
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013429, 0.0013478

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009272, upper bound: 0.0009653
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009263, upper bound: 0.0009727
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007677, 0.0007620
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0019480, 0.0019337
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0012086, 0.0011997
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0022401, 0.0022567
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0019815, 0.0019669
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007505, 0.0007450
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0028429, 0.0028641
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019893, 0.0020041
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0021329, 0.0021487
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0014194, 0.0014089

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009199, upper bound: 0.0009690
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009177, upper bound: 0.0009755
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007557, 0.0007737
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0019177, 0.0019634
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011897, 0.0012181
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0022745, 0.0022215
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0019506, 0.0019971
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007388, 0.0007564
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0028866, 0.0028194
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0020199, 0.0019729
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0021656, 0.0021152
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013972, 0.0014305

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009128, upper bound: 0.0009410
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009010, upper bound: 0.0009512
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007573, 0.0007645
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0019217, 0.0019401
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011922, 0.0012037
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0022475, 0.0022262
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0019547, 0.0019734
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007404, 0.0007475
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0028524, 0.0028253
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019960, 0.0019770
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0021400, 0.0021197
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0014002, 0.0014136

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009458, upper bound: 0.0009733
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009458, upper bound: 0.0009746
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007577, 0.0007642
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0019228, 0.0019392
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011929, 0.0012031
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0022465, 0.0022275
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0019558, 0.0019725
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007408, 0.0007471
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0028511, 0.0028269
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019951, 0.0019782
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0021390, 0.0021209
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0014010, 0.0014129

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009690, upper bound: 0.0010199
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009693, upper bound: 0.0010182
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007146, 0.0007172
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018135, 0.0018200
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011251, 0.0011291
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021083, 0.0021008
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018446, 0.0018512
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006987, 0.0007012
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026758, 0.0026662
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018724, 0.0018657
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020075, 0.0020003
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013213, 0.0013261

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009236, upper bound: 0.0010139
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009271, upper bound: 0.0009928
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007171, 0.0007150
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018197, 0.0018145
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011290, 0.0011257
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021021, 0.0021081
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018510, 0.0018457
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007011, 0.0006991
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026678, 0.0026754
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018668, 0.0018721
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020015, 0.0020072
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013259, 0.0013221

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009621, upper bound: 0.0010283
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009608, upper bound: 0.0010396
time: 0.70 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 4.99 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0010275, upper bound: 0.0009938
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0010449, upper bound: 0.0009811
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0010094, upper bound: 0.0009946
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0010314, upper bound: 0.0009828
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009500, upper bound: 0.0009151
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009500, upper bound: 0.0009151
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0010385, upper bound: 0.0010050
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0010238, upper bound: 0.0010060
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0010478, upper bound: 0.0009959
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0010471, upper bound: 0.0009959
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0010333, upper bound: 0.0009978
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0010314, upper bound: 0.0009981
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009160, upper bound: 0.0008884
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009160, upper bound: 0.0008884
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009243, upper bound: 0.0008831
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009243, upper bound: 0.0008831
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0010222, upper bound: 0.0009519
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0010116, upper bound: 0.0009584
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009913, upper bound: 0.0009271
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009940, upper bound: 0.0009269
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0005558, upper bound: 0.0005191
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0005558, upper bound: 0.0005191
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0005428, upper bound: 0.0005255
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0005428, upper bound: 0.0005255
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009248, upper bound: 0.0008737
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009248, upper bound: 0.0008737
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0010111, upper bound: 0.0009587
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009928, upper bound: 0.0009601
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009429, upper bound: 0.0008770
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009429, upper bound: 0.0008770
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0010208, upper bound: 0.0009567
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0010190, upper bound: 0.0009569
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0010273, upper bound: 0.0009500
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0010099, upper bound: 0.0009589
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0010371, upper bound: 0.0009723
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0010361, upper bound: 0.0009721
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009353, upper bound: 0.0008718
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009231, upper bound: 0.0008787
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009353, upper bound: 0.0008718
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009231, upper bound: 0.0008787
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0010169, upper bound: 0.0009553
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0010096, upper bound: 0.0009639
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0010174, upper bound: 0.0009783
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0010170, upper bound: 0.0009786
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0008789, upper bound: 0.0009230
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0008787, upper bound: 0.0009231
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0008787, upper bound: 0.0009361
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0008770, upper bound: 0.0009423
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0008911, upper bound: 0.0009144
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0008904, upper bound: 0.0009227
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0008779, upper bound: 0.0009308
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0008778, upper bound: 0.0009308
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009149, upper bound: 0.0009540
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009142, upper bound: 0.0009542
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009149, upper bound: 0.0009535
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009142, upper bound: 0.0009540
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0008720, upper bound: 0.0009358
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0008718, upper bound: 0.0009362
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0008737, upper bound: 0.0009248
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0008735, upper bound: 0.0009281
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009410, upper bound: 0.0009868
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009323, upper bound: 0.0010097
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009461, upper bound: 0.0010111
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009412, upper bound: 0.0010174
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0008965, upper bound: 0.0009237
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0008955, upper bound: 0.0009280
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0008995, upper bound: 0.0009281
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0008995, upper bound: 0.0009306
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009443, upper bound: 0.0010226
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009445, upper bound: 0.0010230
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009396, upper bound: 0.0010268
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009397, upper bound: 0.0010275
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0008992, upper bound: 0.0009379
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0008993, upper bound: 0.0009391
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0008989, upper bound: 0.0009382
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0008989, upper bound: 0.0009396
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0008943, upper bound: 0.0009450
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009000, upper bound: 0.0009341
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009270, upper bound: 0.0009629
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009257, upper bound: 0.0009638
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009461, upper bound: 0.0009807
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009350, upper bound: 0.0009923
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009249, upper bound: 0.0009759
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009249, upper bound: 0.0009718
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0008831, upper bound: 0.0009285
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0008831, upper bound: 0.0009251
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0008996, upper bound: 0.0009619
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0008994, upper bound: 0.0009619
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0008995, upper bound: 0.0009344
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0008995, upper bound: 0.0009281
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009183, upper bound: 0.0009483
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009067, upper bound: 0.0009591
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009154, upper bound: 0.0009525
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009066, upper bound: 0.0009638
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009272, upper bound: 0.0009653
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009263, upper bound: 0.0009727
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009199, upper bound: 0.0009690
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009177, upper bound: 0.0009755
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009128, upper bound: 0.0009410
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009010, upper bound: 0.0009512
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009458, upper bound: 0.0009733
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009458, upper bound: 0.0009746
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009690, upper bound: 0.0010199
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009693, upper bound: 0.0010182
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009236, upper bound: 0.0010139
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009271, upper bound: 0.0009928
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009621, upper bound: 0.0010283
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 7, lower bound: -0.0009608, upper bound: 0.0010396

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007073, 0.0007054
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017950, 0.0017901
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011136, 0.0011106
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020738, 0.0020794
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018258, 0.0018209
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006916, 0.0006897
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026319, 0.0026390
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018417, 0.0018467
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019746, 0.0019799
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013079, 0.0013043

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009659, upper bound: 0.0009290
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009682, upper bound: 0.0009290
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0006962, 0.0007163
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017667, 0.0018177
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0010960, 0.0011277
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021058, 0.0020466
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0017970, 0.0018489
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006807, 0.0007003
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026725, 0.0025974
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018701, 0.0018175
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020050, 0.0019487
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0012872, 0.0013244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009817, upper bound: 0.0009202
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009834, upper bound: 0.0009190
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007078, 0.0007050
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017962, 0.0017890
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011143, 0.0011099
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020725, 0.0020808
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018270, 0.0018198
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006920, 0.0006893
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026303, 0.0026408
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018405, 0.0018479
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019734, 0.0019812
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013087, 0.0013035

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0005337, upper bound: 0.0005083
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0005337, upper bound: 0.0005083
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0006967, 0.0007159
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017679, 0.0018168
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0010968, 0.0011272
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021047, 0.0020480
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0017982, 0.0018480
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006811, 0.0007000
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026711, 0.0025992
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018691, 0.0018188
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020040, 0.0019500
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0012881, 0.0013238

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0005394, upper bound: 0.0005041
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0005394, upper bound: 0.0005041
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007053, 0.0007114
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017897, 0.0018053
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011103, 0.0011200
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020913, 0.0020733
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018204, 0.0018362
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006895, 0.0006955
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026541, 0.0026312
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018572, 0.0018412
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019912, 0.0019741
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013040, 0.0013153

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009398, upper bound: 0.0009041
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009314, upper bound: 0.0009050
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007012, 0.0007241
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017794, 0.0018376
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011039, 0.0011401
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021288, 0.0020613
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018099, 0.0018691
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006855, 0.0007080
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027017, 0.0026161
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018905, 0.0018306
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020269, 0.0019627
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0012965, 0.0013389

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008994, upper bound: 0.0008718
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009079, upper bound: 0.0008668
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007138, 0.0007250
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018115, 0.0018398
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011238, 0.0011414
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021314, 0.0020985
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018426, 0.0018714
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006979, 0.0007088
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027050, 0.0026633
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018928, 0.0018636
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020294, 0.0019981
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013199, 0.0013405

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009874, upper bound: 0.0009644
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009967, upper bound: 0.0009558
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007143, 0.0007246
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018126, 0.0018388
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011245, 0.0011408
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021301, 0.0020998
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018437, 0.0018703
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006983, 0.0007084
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027034, 0.0026649
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018917, 0.0018647
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020282, 0.0019993
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013207, 0.0013397

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009709, upper bound: 0.0009648
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009822, upper bound: 0.0009579
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007120, 0.0007219
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018067, 0.0018319
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011209, 0.0011365
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021222, 0.0020930
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018377, 0.0018634
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006961, 0.0007058
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026934, 0.0026563
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018847, 0.0018587
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020207, 0.0019929
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013164, 0.0013348

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009396, upper bound: 0.0009030
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009396, upper bound: 0.0009030
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007122, 0.0007218
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018073, 0.0018317
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011212, 0.0011364
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021219, 0.0020936
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018383, 0.0018631
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006963, 0.0007057
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026930, 0.0026571
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018844, 0.0018593
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020204, 0.0019935
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013168, 0.0013346

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009394, upper bound: 0.0009038
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009394, upper bound: 0.0009038
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007124, 0.0007215
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018077, 0.0018309
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011215, 0.0011359
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021210, 0.0020942
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018388, 0.0018623
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006965, 0.0007054
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026918, 0.0026578
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018836, 0.0018598
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020195, 0.0019940
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013171, 0.0013340

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009762, upper bound: 0.0009535
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009953, upper bound: 0.0009496
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007126, 0.0007214
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018083, 0.0018306
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011219, 0.0011357
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021207, 0.0020948
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018394, 0.0018620
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006967, 0.0007053
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026914, 0.0026586
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018833, 0.0018604
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020192, 0.0019946
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013176, 0.0013338

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009305, upper bound: 0.0009048
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009305, upper bound: 0.0009048
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007021, 0.0006970
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017816, 0.0017687
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011053, 0.0010973
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020490, 0.0020639
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018122, 0.0017991
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006864, 0.0006815
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026004, 0.0026194
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018197, 0.0018329
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019510, 0.0019652
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0012981, 0.0012887

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008996, upper bound: 0.0008712
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008990, upper bound: 0.0008724
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0006982, 0.0007104
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017717, 0.0018027
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0010992, 0.0011184
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020884, 0.0020525
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018022, 0.0018337
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006826, 0.0006945
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026504, 0.0026049
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018546, 0.0018228
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019885, 0.0019543
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0012909, 0.0013135

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009057, upper bound: 0.0008776
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008951, upper bound: 0.0008784
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0006914, 0.0007076
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017546, 0.0017956
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0010886, 0.0011140
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020801, 0.0020326
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0017847, 0.0018264
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006760, 0.0006918
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026399, 0.0025797
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018473, 0.0018051
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019806, 0.0019354
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0012784, 0.0013083

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009066, upper bound: 0.0008663
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009068, upper bound: 0.0008667
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0006869, 0.0007210
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017432, 0.0018296
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0010815, 0.0011351
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021195, 0.0020194
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0017731, 0.0018610
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006716, 0.0007049
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026899, 0.0025628
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018822, 0.0017933
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020181, 0.0019227
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0012701, 0.0013330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009066, upper bound: 0.0008663
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009068, upper bound: 0.0008667
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007042, 0.0007051
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017870, 0.0017894
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011087, 0.0011101
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020729, 0.0020702
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018177, 0.0018201
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006885, 0.0006894
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026308, 0.0026273
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018409, 0.0018385
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019737, 0.0019711
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013021, 0.0013038

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009150, upper bound: 0.0008609
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009150, upper bound: 0.0008609
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007064, 0.0007076
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017926, 0.0017956
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011122, 0.0011140
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020802, 0.0020767
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018234, 0.0018265
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006907, 0.0006918
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026400, 0.0026356
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018474, 0.0018442
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019807, 0.0019773
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013061, 0.0013083

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010020, upper bound: 0.0009473
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009845, upper bound: 0.0009490
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007022, 0.0006941
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017820, 0.0017613
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011055, 0.0010927
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020404, 0.0020643
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018126, 0.0017915
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006865, 0.0006786
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0025895, 0.0026199
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018120, 0.0018333
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019427, 0.0019656
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0012984, 0.0012833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009802, upper bound: 0.0009150
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009642, upper bound: 0.0009164
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0006966, 0.0007074
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017677, 0.0017950
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0010967, 0.0011137
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020795, 0.0020478
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0017980, 0.0018259
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006810, 0.0006916
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026391, 0.0025989
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018467, 0.0018186
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019800, 0.0019498
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0012880, 0.0013079

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009829, upper bound: 0.0009145
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009677, upper bound: 0.0009163
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007003, 0.0006943
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017771, 0.0017619
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011025, 0.0010931
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020411, 0.0020587
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018076, 0.0017921
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006847, 0.0006788
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0025904, 0.0026127
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018126, 0.0018283
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019434, 0.0019602
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0012948, 0.0012837

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009149, upper bound: 0.0008630
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009064, upper bound: 0.0008637
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0006948, 0.0007078
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017631, 0.0017961
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0010939, 0.0011143
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020807, 0.0020425
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0017934, 0.0018270
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006793, 0.0006920
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026407, 0.0025922
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018478, 0.0018139
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019812, 0.0019448
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0012846, 0.0013087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009119, upper bound: 0.0008612
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009119, upper bound: 0.0008612
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007087, 0.0007088
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017983, 0.0017988
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011157, 0.0011160
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020838, 0.0020833
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018292, 0.0018297
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006928, 0.0006930
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026446, 0.0026439
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018506, 0.0018501
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019841, 0.0019836
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013103, 0.0013106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010020, upper bound: 0.0009491
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010018, upper bound: 0.0009495
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007090, 0.0007083
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017991, 0.0017974
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011162, 0.0011151
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020822, 0.0020842
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018300, 0.0018283
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006931, 0.0006925
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026426, 0.0026451
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018492, 0.0018509
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019826, 0.0019844
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013108, 0.0013096

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009838, upper bound: 0.0009510
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009835, upper bound: 0.0009510
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0006914, 0.0007081
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017546, 0.0017969
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0010886, 0.0011148
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020816, 0.0020326
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0017847, 0.0018278
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006760, 0.0006923
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026419, 0.0025797
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018487, 0.0018051
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019821, 0.0019354
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0012784, 0.0013093

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009299, upper bound: 0.0008643
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009297, upper bound: 0.0008644
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0006861, 0.0007217
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017411, 0.0018313
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0010802, 0.0011362
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021215, 0.0020170
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0017710, 0.0018628
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006708, 0.0007056
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026925, 0.0025599
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018841, 0.0017913
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020200, 0.0019205
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0012686, 0.0013343

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009299, upper bound: 0.0008643
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009297, upper bound: 0.0008644
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0006985, 0.0007137
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017725, 0.0018112
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0010997, 0.0011237
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020982, 0.0020533
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018029, 0.0018423
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006829, 0.0006978
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026629, 0.0026060
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018633, 0.0018235
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019978, 0.0019551
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0012915, 0.0013197

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010120, upper bound: 0.0009479
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010120, upper bound: 0.0009478
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0006987, 0.0007136
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017730, 0.0018108
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011000, 0.0011234
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020977, 0.0020539
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018034, 0.0018418
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006831, 0.0006976
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026622, 0.0026067
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018629, 0.0018240
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019973, 0.0019557
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0012918, 0.0013193

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010103, upper bound: 0.0009481
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010104, upper bound: 0.0009480
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0006955, 0.0007168
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017648, 0.0018191
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0010949, 0.0011286
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021073, 0.0020445
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0017951, 0.0018503
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006799, 0.0007008
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026745, 0.0025947
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018715, 0.0018156
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020065, 0.0019467
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0012859, 0.0013254

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010180, upper bound: 0.0009408
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010169, upper bound: 0.0009409
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0006989, 0.0007132
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017735, 0.0018099
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011003, 0.0011229
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020967, 0.0020545
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018039, 0.0018410
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006833, 0.0006973
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026610, 0.0026074
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018620, 0.0018245
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019964, 0.0019562
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0012922, 0.0013187

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009082, upper bound: 0.0008687
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009082, upper bound: 0.0008687
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007126, 0.0007309
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018082, 0.0018548
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011218, 0.0011507
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021487, 0.0020947
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018393, 0.0018866
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006967, 0.0007146
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027270, 0.0026585
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019082, 0.0018603
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020459, 0.0019945
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013175, 0.0013514

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009301, upper bound: 0.0008779
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009301, upper bound: 0.0008779
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007134, 0.0007306
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018105, 0.0018541
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011232, 0.0011503
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021479, 0.0020973
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018415, 0.0018859
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006975, 0.0007143
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027260, 0.0026618
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019075, 0.0018626
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020451, 0.0019970
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013191, 0.0013509

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009296, upper bound: 0.0008778
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009296, upper bound: 0.0008778
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0006905, 0.0007077
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017523, 0.0017960
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0010872, 0.0011142
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020806, 0.0020300
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0017824, 0.0018268
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006751, 0.0006920
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026405, 0.0025763
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018477, 0.0018028
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019810, 0.0019329
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0012768, 0.0013086

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009223, upper bound: 0.0008594
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009220, upper bound: 0.0008594
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0006930, 0.0007042
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017585, 0.0017869
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0010910, 0.0011086
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020701, 0.0020372
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0017887, 0.0018176
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006775, 0.0006885
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026272, 0.0025854
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018384, 0.0018092
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019710, 0.0019397
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0012813, 0.0013020

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009131, upper bound: 0.0008667
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009075, upper bound: 0.0008687
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0006853, 0.0007212
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017391, 0.0018303
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0010789, 0.0011355
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021203, 0.0020147
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0017690, 0.0018617
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006700, 0.0007052
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026909, 0.0025569
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018830, 0.0017892
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020188, 0.0019183
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0012671, 0.0013335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009223, upper bound: 0.0008594
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009220, upper bound: 0.0008594
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0006889, 0.0007177
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017483, 0.0018212
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0010847, 0.0011299
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021097, 0.0020253
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0017783, 0.0018524
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006736, 0.0007017
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026775, 0.0025704
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018736, 0.0017986
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020088, 0.0019284
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0012738, 0.0013269

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009131, upper bound: 0.0008667
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009075, upper bound: 0.0008687
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0006970, 0.0007195
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017687, 0.0018258
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0010973, 0.0011327
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021151, 0.0020490
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0017991, 0.0018571
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006815, 0.0007034
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026843, 0.0026005
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018784, 0.0018197
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020139, 0.0019510
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0012887, 0.0013303

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009253, upper bound: 0.0008604
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009253, upper bound: 0.0008604
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007008, 0.0007160
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017785, 0.0018171
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011034, 0.0011273
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021050, 0.0020603
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018090, 0.0018483
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006852, 0.0007001
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026715, 0.0026147
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018694, 0.0018297
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020043, 0.0019617
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0012958, 0.0013239

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009994, upper bound: 0.0009550
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010008, upper bound: 0.0009550
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007147, 0.0007335
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018137, 0.0018613
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011252, 0.0011548
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021563, 0.0021010
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018448, 0.0018933
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006988, 0.0007171
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027366, 0.0026665
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019149, 0.0018659
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020531, 0.0020005
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013214, 0.0013562

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009280, upper bound: 0.0008782
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009280, upper bound: 0.0008782
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007156, 0.0007332
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018159, 0.0018607
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011266, 0.0011544
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021555, 0.0021037
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018471, 0.0018926
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006996, 0.0007169
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0027356, 0.0026698
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0019142, 0.0018682
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020524, 0.0020030
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013231, 0.0013557

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009279, upper bound: 0.0008780
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009279, upper bound: 0.0008780
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007043, 0.0006891
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017873, 0.0017488
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011089, 0.0010850
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020259, 0.0020706
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018180, 0.0017788
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006886, 0.0006738
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0025711, 0.0026278
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0017992, 0.0018388
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019290, 0.0019715
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013023, 0.0012742

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008666, upper bound: 0.0009105
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008667, upper bound: 0.0009101
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007044, 0.0006889
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017876, 0.0017483
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011091, 0.0010847
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020253, 0.0020709
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018183, 0.0017783
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006887, 0.0006736
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0025704, 0.0026282
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0017986, 0.0018391
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019284, 0.0019718
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013025, 0.0012738

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008662, upper bound: 0.0009105
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008664, upper bound: 0.0009103
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007095, 0.0006885
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018005, 0.0017471
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011170, 0.0010839
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020240, 0.0020858
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018314, 0.0017771
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006937, 0.0006731
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0025687, 0.0026471
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0017974, 0.0018523
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019271, 0.0019860
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013119, 0.0012730

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008620, upper bound: 0.0009190
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008618, upper bound: 0.0009196
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007100, 0.0006880
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018017, 0.0017460
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011178, 0.0010832
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020226, 0.0020872
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018327, 0.0017760
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006942, 0.0006727
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0025670, 0.0026490
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0017962, 0.0018536
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019259, 0.0019874
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013128, 0.0012721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008604, upper bound: 0.0009253
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008601, upper bound: 0.0009255
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0006967, 0.0007041
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017681, 0.0017868
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0010969, 0.0011085
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020699, 0.0020482
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0017984, 0.0018175
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006812, 0.0006884
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026270, 0.0025995
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018383, 0.0018190
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019709, 0.0019502
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0012882, 0.0013019

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008750, upper bound: 0.0008970
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008739, upper bound: 0.0008984
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0006973, 0.0007038
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017694, 0.0017860
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0010978, 0.0011080
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020690, 0.0020498
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0017998, 0.0018167
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006817, 0.0006881
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026258, 0.0026015
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018374, 0.0018204
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019700, 0.0019517
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0012892, 0.0013013

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008742, upper bound: 0.0009053
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008734, upper bound: 0.0009065
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0006980, 0.0006986
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017712, 0.0017729
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0010989, 0.0010999
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020538, 0.0020519
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018016, 0.0018033
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006824, 0.0006830
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026065, 0.0026041
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018239, 0.0018222
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019555, 0.0019537
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0012905, 0.0012917

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008677, upper bound: 0.0009128
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008669, upper bound: 0.0009209
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0006982, 0.0006976
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017717, 0.0017703
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0010992, 0.0010983
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020508, 0.0020524
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018021, 0.0018007
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006826, 0.0006820
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026027, 0.0026048
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018213, 0.0018227
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019527, 0.0019542
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0012909, 0.0012899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008612, upper bound: 0.0009119
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008609, upper bound: 0.0009150
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007134, 0.0007092
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018104, 0.0017996
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011232, 0.0011165
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020848, 0.0020973
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018415, 0.0018305
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006975, 0.0006933
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026458, 0.0026617
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018514, 0.0018625
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019850, 0.0019969
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013191, 0.0013112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009048, upper bound: 0.0009368
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009038, upper bound: 0.0009439
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007134, 0.0007090
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018104, 0.0017991
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011232, 0.0011162
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020842, 0.0020973
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018415, 0.0018300
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006975, 0.0006932
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026451, 0.0026617
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018509, 0.0018625
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019845, 0.0019969
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013191, 0.0013109

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009041, upper bound: 0.0009378
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009030, upper bound: 0.0009441
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007136, 0.0007083
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018108, 0.0017973
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011234, 0.0011151
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020821, 0.0020977
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018419, 0.0018282
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006977, 0.0006925
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026425, 0.0026623
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018491, 0.0018630
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019825, 0.0019974
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013194, 0.0013096

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008665, upper bound: 0.0009108
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008718, upper bound: 0.0009022
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007136, 0.0007081
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018108, 0.0017968
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011234, 0.0011148
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020815, 0.0020977
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018419, 0.0018277
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006977, 0.0006923
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026418, 0.0026623
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018486, 0.0018630
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019820, 0.0019974
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013194, 0.0013092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009042, upper bound: 0.0009373
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009031, upper bound: 0.0009438
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007089, 0.0006879
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017990, 0.0017456
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011161, 0.0010830
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020222, 0.0020840
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018299, 0.0017756
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006931, 0.0006725
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0025665, 0.0026449
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0017959, 0.0018508
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019255, 0.0019843
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013108, 0.0012719

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008596, upper bound: 0.0009226
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008596, upper bound: 0.0009229
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007091, 0.0006878
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017994, 0.0017454
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011163, 0.0010829
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020220, 0.0020845
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018303, 0.0017754
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006933, 0.0006725
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0025661, 0.0026455
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0017957, 0.0018512
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019252, 0.0019848
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013111, 0.0012717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008618, upper bound: 0.0009219
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008601, upper bound: 0.0009264
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0006983, 0.0006994
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017720, 0.0017748
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0010993, 0.0011011
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020560, 0.0020527
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018024, 0.0018053
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006827, 0.0006838
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026094, 0.0026052
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018259, 0.0018230
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019577, 0.0019545
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0012911, 0.0012932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008637, upper bound: 0.0009064
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008630, upper bound: 0.0009149
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0006984, 0.0006994
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0017723, 0.0017747
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0010995, 0.0011011
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020560, 0.0020531
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018027, 0.0018052
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006828, 0.0006838
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026093, 0.0026057
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018258, 0.0018233
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019576, 0.0019549
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0012913, 0.0012931

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008635, upper bound: 0.0009106
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008628, upper bound: 0.0009181
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007121, 0.0006879
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018070, 0.0017457
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011211, 0.0010831
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020223, 0.0020934
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018381, 0.0017757
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006962, 0.0006726
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0025666, 0.0026567
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0017960, 0.0018591
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019256, 0.0019932
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013166, 0.0012720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009251, upper bound: 0.0009690
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009206, upper bound: 0.0009696
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007155, 0.0006852
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018156, 0.0017387
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011264, 0.0010787
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020142, 0.0021033
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018468, 0.0017685
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006995, 0.0006699
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0025563, 0.0026694
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0017888, 0.0018679
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019178, 0.0020027
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013229, 0.0012668

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008663, upper bound: 0.0009185
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008663, upper bound: 0.0009205
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007300, 0.0007015
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018525, 0.0017802
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011493, 0.0011045
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020623, 0.0021460
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018843, 0.0018108
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007137, 0.0006859
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026174, 0.0027235
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018315, 0.0019058
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019637, 0.0020433
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013497, 0.0012971

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009252, upper bound: 0.0009694
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009172, upper bound: 0.0009888
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007301, 0.0007013
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018526, 0.0017796
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011494, 0.0011041
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020616, 0.0021462
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018844, 0.0018102
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007138, 0.0006856
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026164, 0.0027238
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018309, 0.0019060
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019630, 0.0020435
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013498, 0.0012967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008781, upper bound: 0.0009256
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008780, upper bound: 0.0009288
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007155, 0.0007135
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018157, 0.0018107
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011265, 0.0011234
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020976, 0.0021034
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018469, 0.0018418
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006996, 0.0006976
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026621, 0.0026695
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018628, 0.0018680
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019973, 0.0020028
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013230, 0.0013193

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008750, upper bound: 0.0008916
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008647, upper bound: 0.0009021
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007156, 0.0007133
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018160, 0.0018100
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011267, 0.0011230
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020968, 0.0021038
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018472, 0.0018411
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0006997, 0.0006974
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026612, 0.0026700
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018622, 0.0018683
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019965, 0.0020031
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013232, 0.0013188

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008739, upper bound: 0.0008932
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008642, upper bound: 0.0009065
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007175, 0.0007166
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018207, 0.0018184
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011296, 0.0011282
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021066, 0.0021092
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018519, 0.0018496
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007015, 0.0007006
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026735, 0.0026768
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018708, 0.0018731
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020058, 0.0020083
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013266, 0.0013249

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008779, upper bound: 0.0008956
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008684, upper bound: 0.0009060
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007182, 0.0007164
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018227, 0.0018179
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011308, 0.0011278
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0021059, 0.0021115
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018539, 0.0018491
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007022, 0.0007004
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026727, 0.0026797
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018702, 0.0018751
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0020052, 0.0020104
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013280, 0.0013245

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008779, upper bound: 0.0008962
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008688, upper bound: 0.0009088
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007294, 0.0007012
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018509, 0.0017795
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011483, 0.0011040
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020614, 0.0021442
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018827, 0.0018100
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007131, 0.0006856
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026162, 0.0027212
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018307, 0.0019042
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019628, 0.0020416
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013486, 0.0012965

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009232, upper bound: 0.0009814
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009145, upper bound: 0.0010008
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007304, 0.0007011
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018535, 0.0017792
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011499, 0.0011038
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020611, 0.0021472
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018853, 0.0018097
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007141, 0.0006855
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026158, 0.0027251
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018304, 0.0019069
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019625, 0.0020445
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013505, 0.0012963

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009233, upper bound: 0.0009815
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009154, upper bound: 0.0010009
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027842, -0.0015775, -0.0027842, -0.0015775, -0.0007294, 0.0007009
1: -0.0113759, -0.0083137, -0.0113759, -0.0083137, -0.0018509, 0.0017787
2: 0.0279724, 0.0298721, 0.0279724, 0.0298721, -0.0011483, 0.0011035
3: 0.0037837, 0.0073310, 0.0037837, 0.0073310, -0.0020605, 0.0021442
4: -0.0104642, -0.0073495, -0.0104642, -0.0073495, -0.0018827, 0.0018092
5: 0.0097746, 0.0109544, 0.0097746, 0.0109544, -0.0007131, 0.0006853
6: 0.0051901, 0.0096921, 0.0051901, 0.0096921, -0.0026151, 0.0027212
7: 0.9816911, 0.9848413, 0.9816911, 0.9848413, -0.0018299, 0.0019042
8: -0.0061943, -0.0028167, -0.0061943, -0.0028167, -0.0019620, 0.0020416
9: -0.0031390, -0.0009079, -0.0031390, -0.0009079, -0.0013486, 0.0012960

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 3.04 + 597.96 = 601.00 seconds
