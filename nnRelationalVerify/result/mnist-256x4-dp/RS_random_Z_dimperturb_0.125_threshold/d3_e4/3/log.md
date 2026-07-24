## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00049488


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9877946, 0.9898415, 0.9877946, 0.9898415, -0.0013072, 0.0013072)
1: (-0.0043052, -0.0037952, -0.0043052, -0.0037952, -0.0003257, 0.0003257)
2: (0.0100586, 0.0127613, 0.0100586, 0.0127613, -0.0017262, 0.0017262)
3: (-0.0070815, -0.0058513, -0.0070815, -0.0058513, -0.0007857, 0.0007857)
4: (0.0024747, 0.0029978, 0.0024747, 0.0029978, -0.0003341, 0.0003341)
5: (0.0116105, 0.0150099, 0.0116105, 0.0150099, -0.0021710, 0.0021710)
6: (-0.0022688, -0.0014060, -0.0022688, -0.0014060, -0.0005510, 0.0005510)
7: (-0.0090078, -0.0067755, -0.0090078, -0.0067755, -0.0014257, 0.0014257)
8: (-0.0043013, -0.0031273, -0.0043013, -0.0031273, -0.0007498, 0.0007498)
9: (0.0017624, 0.0031237, 0.0017624, 0.0031237, -0.0008694, 0.0008694)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.96 + 1.39 = 3.35 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0005930, upper bound: 0.0005930

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005369, upper bound: 0.0005370
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005369, upper bound: 0.0005370
time: 0.57 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.15 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.15
Output dim: 0, lower bound: -0.0005369, upper bound: 0.0005370
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.15
Output dim: 0, lower bound: -0.0005369, upper bound: 0.0005370

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.9877946, 0.9898415, 0.9877946, 0.9898415, -0.0012971, 0.0012869
1: -0.0043052, -0.0037952, -0.0043052, -0.0037952, -0.0003232, 0.0003207
2: 0.0100586, 0.0127613, 0.0100586, 0.0127613, -0.0016993, 0.0017129
3: -0.0070815, -0.0058513, -0.0070815, -0.0058513, -0.0007796, 0.0007734
4: 0.0024747, 0.0029978, 0.0024747, 0.0029978, -0.0003289, 0.0003315
5: 0.0116105, 0.0150099, 0.0116105, 0.0150099, -0.0021373, 0.0021543
6: -0.0022688, -0.0014060, -0.0022688, -0.0014060, -0.0005468, 0.0005425
7: -0.0090078, -0.0067755, -0.0090078, -0.0067755, -0.0014147, 0.0014035
8: -0.0043013, -0.0031273, -0.0043013, -0.0031273, -0.0007440, 0.0007381
9: 0.0017624, 0.0031237, 0.0017624, 0.0031237, -0.0008559, 0.0008627

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004894, upper bound: 0.0004887
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004886, upper bound: 0.0004895
time: 0.53 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.9877946, 0.9898415, 0.9877946, 0.9898415, -0.0013072, 0.0012971
1: -0.0043052, -0.0037952, -0.0043052, -0.0037952, -0.0003257, 0.0003232
2: 0.0100586, 0.0127613, 0.0100586, 0.0127613, -0.0017129, 0.0017262
3: -0.0070815, -0.0058513, -0.0070815, -0.0058513, -0.0007857, 0.0007796
4: 0.0024747, 0.0029978, 0.0024747, 0.0029978, -0.0003315, 0.0003341
5: 0.0116105, 0.0150099, 0.0116105, 0.0150099, -0.0021543, 0.0021710
6: -0.0022688, -0.0014060, -0.0022688, -0.0014060, -0.0005510, 0.0005468
7: -0.0090078, -0.0067755, -0.0090078, -0.0067755, -0.0014257, 0.0014147
8: -0.0043013, -0.0031273, -0.0043013, -0.0031273, -0.0007498, 0.0007440
9: 0.0017624, 0.0031237, 0.0017624, 0.0031237, -0.0008627, 0.0008694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005296, upper bound: 0.0005296
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005296, upper bound: 0.0005296
time: 0.52 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.69 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.69
Output dim: 0, lower bound: -0.0004894, upper bound: 0.0004887
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.69
Output dim: 0, lower bound: -0.0004886, upper bound: 0.0004895
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.69
Output dim: 0, lower bound: -0.0005296, upper bound: 0.0005296
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.69
Output dim: 0, lower bound: -0.0005296, upper bound: 0.0005296

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9877946, 0.9898415, 0.9877946, 0.9898415, -0.0013025, 0.0012920
1: -0.0043052, -0.0037952, -0.0043052, -0.0037952, -0.0003245, 0.0003219
2: 0.0100586, 0.0127613, 0.0100586, 0.0127613, -0.0017061, 0.0017199
3: -0.0070815, -0.0058513, -0.0070815, -0.0058513, -0.0007828, 0.0007765
4: 0.0024747, 0.0029978, 0.0024747, 0.0029978, -0.0003302, 0.0003329
5: 0.0116105, 0.0150099, 0.0116105, 0.0150099, -0.0021458, 0.0021632
6: -0.0022688, -0.0014060, -0.0022688, -0.0014060, -0.0005490, 0.0005446
7: -0.0090078, -0.0067755, -0.0090078, -0.0067755, -0.0014205, 0.0014091
8: -0.0043013, -0.0031273, -0.0043013, -0.0031273, -0.0007471, 0.0007410
9: 0.0017624, 0.0031237, 0.0017624, 0.0031237, -0.0008593, 0.0008662

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005155, upper bound: 0.0004960
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004960, upper bound: 0.0005156
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9877946, 0.9898415, 0.9877946, 0.9898415, -0.0013021, 0.0012924
1: -0.0043052, -0.0037952, -0.0043052, -0.0037952, -0.0003244, 0.0003220
2: 0.0100586, 0.0127613, 0.0100586, 0.0127613, -0.0017066, 0.0017194
3: -0.0070815, -0.0058513, -0.0070815, -0.0058513, -0.0007826, 0.0007768
4: 0.0024747, 0.0029978, 0.0024747, 0.0029978, -0.0003303, 0.0003328
5: 0.0116105, 0.0150099, 0.0116105, 0.0150099, -0.0021465, 0.0021625
6: -0.0022688, -0.0014060, -0.0022688, -0.0014060, -0.0005489, 0.0005448
7: -0.0090078, -0.0067755, -0.0090078, -0.0067755, -0.0014201, 0.0014096
8: -0.0043013, -0.0031273, -0.0043013, -0.0031273, -0.0007468, 0.0007413
9: 0.0017624, 0.0031237, 0.0017624, 0.0031237, -0.0008596, 0.0008660

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005155, upper bound: 0.0004960
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004960, upper bound: 0.0005156
time: 0.53 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.77 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.77
Output dim: 0, lower bound: -0.0005155, upper bound: 0.0004960
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.77
Output dim: 0, lower bound: -0.0004960, upper bound: 0.0005156
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.77
Output dim: 0, lower bound: -0.0005155, upper bound: 0.0004960
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.77
Output dim: 0, lower bound: -0.0004960, upper bound: 0.0005156

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9877946, 0.9898415, 0.9877946, 0.9898415, -0.0012690, 0.0012489
1: -0.0043052, -0.0037952, -0.0043052, -0.0037952, -0.0003162, 0.0003112
2: 0.0100586, 0.0127613, 0.0100586, 0.0127613, -0.0016492, 0.0016757
3: -0.0070815, -0.0058513, -0.0070815, -0.0058513, -0.0007627, 0.0007506
4: 0.0024747, 0.0029978, 0.0024747, 0.0029978, -0.0003192, 0.0003243
5: 0.0116105, 0.0150099, 0.0116105, 0.0150099, -0.0020743, 0.0021076
6: -0.0022688, -0.0014060, -0.0022688, -0.0014060, -0.0005349, 0.0005265
7: -0.0090078, -0.0067755, -0.0090078, -0.0067755, -0.0013840, 0.0013621
8: -0.0043013, -0.0031273, -0.0043013, -0.0031273, -0.0007278, 0.0007163
9: 0.0017624, 0.0031237, 0.0017624, 0.0031237, -0.0008306, 0.0008440

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004687, upper bound: 0.0004480
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004670, upper bound: 0.0004509
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9877946, 0.9898415, 0.9877946, 0.9898415, -0.0012599, 0.0012584
1: -0.0043052, -0.0037952, -0.0043052, -0.0037952, -0.0003139, 0.0003136
2: 0.0100586, 0.0127613, 0.0100586, 0.0127613, -0.0016617, 0.0016636
3: -0.0070815, -0.0058513, -0.0070815, -0.0058513, -0.0007572, 0.0007563
4: 0.0024747, 0.0029978, 0.0024747, 0.0029978, -0.0003216, 0.0003220
5: 0.0116105, 0.0150099, 0.0116105, 0.0150099, -0.0020899, 0.0020924
6: -0.0022688, -0.0014060, -0.0022688, -0.0014060, -0.0005311, 0.0005305
7: -0.0090078, -0.0067755, -0.0090078, -0.0067755, -0.0013741, 0.0013724
8: -0.0043013, -0.0031273, -0.0043013, -0.0031273, -0.0007226, 0.0007218
9: 0.0017624, 0.0031237, 0.0017624, 0.0031237, -0.0008369, 0.0008379

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004595, upper bound: 0.0005035
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004837, upper bound: 0.0004654
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9877946, 0.9898415, 0.9877946, 0.9898415, -0.0012689, 0.0012493
1: -0.0043052, -0.0037952, -0.0043052, -0.0037952, -0.0003162, 0.0003113
2: 0.0100586, 0.0127613, 0.0100586, 0.0127613, -0.0016497, 0.0016756
3: -0.0070815, -0.0058513, -0.0070815, -0.0058513, -0.0007626, 0.0007509
4: 0.0024747, 0.0029978, 0.0024747, 0.0029978, -0.0003193, 0.0003243
5: 0.0116105, 0.0150099, 0.0116105, 0.0150099, -0.0020749, 0.0021074
6: -0.0022688, -0.0014060, -0.0022688, -0.0014060, -0.0005349, 0.0005266
7: -0.0090078, -0.0067755, -0.0090078, -0.0067755, -0.0013839, 0.0013626
8: -0.0043013, -0.0031273, -0.0043013, -0.0031273, -0.0007278, 0.0007166
9: 0.0017624, 0.0031237, 0.0017624, 0.0031237, -0.0008309, 0.0008439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004654, upper bound: 0.0004837
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005035, upper bound: 0.0004595
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9877946, 0.9898415, 0.9877946, 0.9898415, -0.0012595, 0.0012585
1: -0.0043052, -0.0037952, -0.0043052, -0.0037952, -0.0003138, 0.0003136
2: 0.0100586, 0.0127613, 0.0100586, 0.0127613, -0.0016618, 0.0016631
3: -0.0070815, -0.0058513, -0.0070815, -0.0058513, -0.0007570, 0.0007564
4: 0.0024747, 0.0029978, 0.0024747, 0.0029978, -0.0003216, 0.0003219
5: 0.0116105, 0.0150099, 0.0116105, 0.0150099, -0.0020901, 0.0020917
6: -0.0022688, -0.0014060, -0.0022688, -0.0014060, -0.0005309, 0.0005305
7: -0.0090078, -0.0067755, -0.0090078, -0.0067755, -0.0013736, 0.0013725
8: -0.0043013, -0.0031273, -0.0043013, -0.0031273, -0.0007224, 0.0007218
9: 0.0017624, 0.0031237, 0.0017624, 0.0031237, -0.0008370, 0.0008376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004508, upper bound: 0.0004670
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004480, upper bound: 0.0004688
time: 0.58 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.80 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.80
Output dim: 0, lower bound: -0.0004687, upper bound: 0.0004480
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.80
Output dim: 0, lower bound: -0.0004670, upper bound: 0.0004509
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 0, lower bound: -0.0004595, upper bound: 0.0005035
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.80
Output dim: 0, lower bound: -0.0004837, upper bound: 0.0004654
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.80
Output dim: 0, lower bound: -0.0004654, upper bound: 0.0004837
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 0, lower bound: -0.0005035, upper bound: 0.0004595
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.80
Output dim: 0, lower bound: -0.0004508, upper bound: 0.0004670
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.80
Output dim: 0, lower bound: -0.0004480, upper bound: 0.0004688

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9877946, 0.9898415, 0.9877946, 0.9898415, -0.0011877, 0.0012074
1: -0.0043052, -0.0037952, -0.0043052, -0.0037952, -0.0002960, 0.0003009
2: 0.0100586, 0.0127613, 0.0100586, 0.0127613, -0.0015944, 0.0015684
3: -0.0070815, -0.0058513, -0.0070815, -0.0058513, -0.0007139, 0.0007257
4: 0.0024747, 0.0029978, 0.0024747, 0.0029978, -0.0003086, 0.0003036
5: 0.0116105, 0.0150099, 0.0116105, 0.0150099, -0.0020053, 0.0019726
6: -0.0022688, -0.0014060, -0.0022688, -0.0014060, -0.0005007, 0.0005090
7: -0.0090078, -0.0067755, -0.0090078, -0.0067755, -0.0012954, 0.0013169
8: -0.0043013, -0.0031273, -0.0043013, -0.0031273, -0.0006812, 0.0006925
9: 0.0017624, 0.0031237, 0.0017624, 0.0031237, -0.0008030, 0.0007899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004440, upper bound: 0.0004872
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004428, upper bound: 0.0004872
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9877946, 0.9898415, 0.9877946, 0.9898415, -0.0012171, 0.0011780
1: -0.0043052, -0.0037952, -0.0043052, -0.0037952, -0.0003033, 0.0002935
2: 0.0100586, 0.0127613, 0.0100586, 0.0127613, -0.0015556, 0.0016072
3: -0.0070815, -0.0058513, -0.0070815, -0.0058513, -0.0007315, 0.0007080
4: 0.0024747, 0.0029978, 0.0024747, 0.0029978, -0.0003011, 0.0003111
5: 0.0116105, 0.0150099, 0.0116105, 0.0150099, -0.0019565, 0.0020214
6: -0.0022688, -0.0014060, -0.0022688, -0.0014060, -0.0005131, 0.0004966
7: -0.0090078, -0.0067755, -0.0090078, -0.0067755, -0.0013274, 0.0012848
8: -0.0043013, -0.0031273, -0.0043013, -0.0031273, -0.0006981, 0.0006757
9: 0.0017624, 0.0031237, 0.0017624, 0.0031237, -0.0007835, 0.0008095

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004802, upper bound: 0.0004044
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004575, upper bound: 0.0004386
time: 0.54 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.77 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.77
Output dim: 0, lower bound: -0.0004440, upper bound: 0.0004872
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.77
Output dim: 0, lower bound: -0.0004428, upper bound: 0.0004872
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.77
Output dim: 0, lower bound: -0.0004802, upper bound: 0.0004044
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.77
Output dim: 0, lower bound: -0.0004575, upper bound: 0.0004386

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.35 + 28.88 = 32.22 seconds
