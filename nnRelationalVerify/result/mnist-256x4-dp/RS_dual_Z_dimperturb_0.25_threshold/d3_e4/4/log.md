## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00221263


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0060547, 0.0095268, 0.0060547, 0.0095268, -0.0032140, 0.0032140)
1: (0.0016576, 0.0026344, 0.0016576, 0.0026344, -0.0009479, 0.0009479)
2: (0.0089833, 0.0110124, 0.0089833, 0.0110124, -0.0019671, 0.0019671)
3: (-0.0050220, -0.0027756, -0.0050220, -0.0027756, -0.0019561, 0.0019561)
4: (-0.0005241, 0.0013996, -0.0005241, 0.0013996, -0.0015579, 0.0015579)
5: (0.0026192, 0.0045622, 0.0026192, 0.0045622, -0.0017289, 0.0017289)
6: (-0.0113972, -0.0041987, -0.0113972, -0.0041987, -0.0059028, 0.0059028)
7: (0.0025876, 0.0127445, 0.0025876, 0.0127445, -0.0083132, 0.0083132)
8: (0.9914410, 0.9982397, 0.9914410, 0.9982397, -0.0054809, 0.0054809)
9: (-0.0142455, -0.0079315, -0.0142455, -0.0079315, -0.0051198, 0.0051198)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.99 + 1.53 = 3.52 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0024914, upper bound: 0.0024914

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023221, upper bound: 0.0024103
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024103, upper bound: 0.0023222
time: 0.74 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.68 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.68
Output dim: 8, lower bound: -0.0023221, upper bound: 0.0024103
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.68
Output dim: 8, lower bound: -0.0024103, upper bound: 0.0023222

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060547, 0.0095268, 0.0060547, 0.0095268, -0.0030504, 0.0030591
1: 0.0016576, 0.0026344, 0.0016576, 0.0026344, -0.0009267, 0.0009279
2: 0.0089833, 0.0110124, 0.0089833, 0.0110124, -0.0018814, 0.0018766
3: -0.0050220, -0.0027756, -0.0050220, -0.0027756, -0.0018712, 0.0018662
4: -0.0005241, 0.0013996, -0.0005241, 0.0013996, -0.0014576, 0.0014630
5: 0.0026192, 0.0045622, 0.0026192, 0.0045622, -0.0016381, 0.0016331
6: -0.0113972, -0.0041987, -0.0113972, -0.0041987, -0.0055427, 0.0055226
7: 0.0025876, 0.0127445, 0.0025876, 0.0127445, -0.0078041, 0.0078315
8: 0.9914410, 0.9982397, 0.9914410, 0.9982397, -0.0051161, 0.0051354
9: -0.0142455, -0.0079315, -0.0142455, -0.0079315, -0.0048098, 0.0047923

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022164, upper bound: 0.0023046
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022164, upper bound: 0.0023046
time: 0.59 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060547, 0.0095268, 0.0060547, 0.0095268, -0.0030591, 0.0030504
1: 0.0016576, 0.0026344, 0.0016576, 0.0026344, -0.0009279, 0.0009267
2: 0.0089833, 0.0110124, 0.0089833, 0.0110124, -0.0018766, 0.0018814
3: -0.0050220, -0.0027756, -0.0050220, -0.0027756, -0.0018662, 0.0018712
4: -0.0005241, 0.0013996, -0.0005241, 0.0013996, -0.0014630, 0.0014576
5: 0.0026192, 0.0045622, 0.0026192, 0.0045622, -0.0016331, 0.0016381
6: -0.0113972, -0.0041987, -0.0113972, -0.0041987, -0.0055226, 0.0055427
7: 0.0025876, 0.0127445, 0.0025876, 0.0127445, -0.0078315, 0.0078041
8: 0.9914410, 0.9982397, 0.9914410, 0.9982397, -0.0051354, 0.0051161
9: -0.0142455, -0.0079315, -0.0142455, -0.0079315, -0.0047923, 0.0048098

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023046, upper bound: 0.0022164
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023046, upper bound: 0.0022164
time: 0.61 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.19 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.19
Output dim: 8, lower bound: -0.0022164, upper bound: 0.0023046
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.19
Output dim: 8, lower bound: -0.0022164, upper bound: 0.0023046
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.19
Output dim: 8, lower bound: -0.0023046, upper bound: 0.0022164
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.19
Output dim: 8, lower bound: -0.0023046, upper bound: 0.0022164

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060547, 0.0095268, 0.0060547, 0.0095268, -0.0029985, 0.0030041
1: 0.0016576, 0.0026344, 0.0016576, 0.0026344, -0.0009196, 0.0009205
2: 0.0089833, 0.0110124, 0.0089833, 0.0110124, -0.0018510, 0.0018479
3: -0.0050220, -0.0027756, -0.0050220, -0.0027756, -0.0018424, 0.0018392
4: -0.0005241, 0.0013996, -0.0005241, 0.0013996, -0.0014259, 0.0014294
5: 0.0026192, 0.0045622, 0.0026192, 0.0045622, -0.0016059, 0.0016026
6: -0.0113972, -0.0041987, -0.0113972, -0.0041987, -0.0054148, 0.0054018
7: 0.0025876, 0.0127445, 0.0025876, 0.0127445, -0.0076425, 0.0076602
8: 0.9914410, 0.9982397, 0.9914410, 0.9982397, -0.0050002, 0.0050127
9: -0.0142455, -0.0079315, -0.0142455, -0.0079315, -0.0047001, 0.0046887

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020019, upper bound: 0.0020741
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020019, upper bound: 0.0020741
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060547, 0.0095268, 0.0060547, 0.0095268, -0.0029962, 0.0030071
1: 0.0016576, 0.0026344, 0.0016576, 0.0026344, -0.0009193, 0.0009209
2: 0.0089833, 0.0110124, 0.0089833, 0.0110124, -0.0018527, 0.0018466
3: -0.0050220, -0.0027756, -0.0050220, -0.0027756, -0.0018441, 0.0018379
4: -0.0005241, 0.0013996, -0.0005241, 0.0013996, -0.0014245, 0.0014313
5: 0.0026192, 0.0045622, 0.0026192, 0.0045622, -0.0016077, 0.0016013
6: -0.0113972, -0.0041987, -0.0113972, -0.0041987, -0.0054219, 0.0053965
7: 0.0025876, 0.0127445, 0.0025876, 0.0127445, -0.0076352, 0.0076699
8: 0.9914410, 0.9982397, 0.9914410, 0.9982397, -0.0049951, 0.0050195
9: -0.0142455, -0.0079315, -0.0142455, -0.0079315, -0.0047063, 0.0046841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020019, upper bound: 0.0020741
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020019, upper bound: 0.0020741
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060547, 0.0095268, 0.0060547, 0.0095268, -0.0030071, 0.0029962
1: 0.0016576, 0.0026344, 0.0016576, 0.0026344, -0.0009209, 0.0009193
2: 0.0089833, 0.0110124, 0.0089833, 0.0110124, -0.0018466, 0.0018527
3: -0.0050220, -0.0027756, -0.0050220, -0.0027756, -0.0018379, 0.0018441
4: -0.0005241, 0.0013996, -0.0005241, 0.0013996, -0.0014313, 0.0014245
5: 0.0026192, 0.0045622, 0.0026192, 0.0045622, -0.0016013, 0.0016077
6: -0.0113972, -0.0041987, -0.0113972, -0.0041987, -0.0053965, 0.0054219
7: 0.0025876, 0.0127445, 0.0025876, 0.0127445, -0.0076699, 0.0076352
8: 0.9914410, 0.9982397, 0.9914410, 0.9982397, -0.0050195, 0.0049951
9: -0.0142455, -0.0079315, -0.0142455, -0.0079315, -0.0046841, 0.0047063

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020741, upper bound: 0.0020019
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020741, upper bound: 0.0020019
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060547, 0.0095268, 0.0060547, 0.0095268, -0.0030041, 0.0029985
1: 0.0016576, 0.0026344, 0.0016576, 0.0026344, -0.0009205, 0.0009196
2: 0.0089833, 0.0110124, 0.0089833, 0.0110124, -0.0018479, 0.0018510
3: -0.0050220, -0.0027756, -0.0050220, -0.0027756, -0.0018392, 0.0018424
4: -0.0005241, 0.0013996, -0.0005241, 0.0013996, -0.0014294, 0.0014259
5: 0.0026192, 0.0045622, 0.0026192, 0.0045622, -0.0016026, 0.0016059
6: -0.0113972, -0.0041987, -0.0113972, -0.0041987, -0.0054018, 0.0054148
7: 0.0025876, 0.0127445, 0.0025876, 0.0127445, -0.0076602, 0.0076425
8: 0.9914410, 0.9982397, 0.9914410, 0.9982397, -0.0050127, 0.0050002
9: -0.0142455, -0.0079315, -0.0142455, -0.0079315, -0.0046887, 0.0047001

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020741, upper bound: 0.0020019
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020741, upper bound: 0.0020019
time: 0.67 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.22 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.22
Output dim: 8, lower bound: -0.0020019, upper bound: 0.0020741
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.22
Output dim: 8, lower bound: -0.0020019, upper bound: 0.0020741
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.22
Output dim: 8, lower bound: -0.0020019, upper bound: 0.0020741
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.22
Output dim: 8, lower bound: -0.0020019, upper bound: 0.0020741
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.22
Output dim: 8, lower bound: -0.0020741, upper bound: 0.0020019
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.22
Output dim: 8, lower bound: -0.0020741, upper bound: 0.0020019
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.22
Output dim: 8, lower bound: -0.0020741, upper bound: 0.0020019
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.22
Output dim: 8, lower bound: -0.0020741, upper bound: 0.0020019

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.52 + 20.47 = 23.99 seconds
