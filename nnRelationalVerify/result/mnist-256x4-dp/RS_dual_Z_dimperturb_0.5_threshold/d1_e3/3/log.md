## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.01370172


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093)
1: (0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0100059, 0.0100059)
2: (0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0071291, 0.0071291)
3: (0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0091684, 0.0091684)
4: (-0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0093412, 0.0093412)
5: (0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0118050, 0.0118050)
6: (0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0089926, 0.0089926)
7: (-0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0100511, 0.0100511)
8: (0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0093038, 0.0093038)
9: (0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0365864, 0.0365864)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.23 + 1.53 = 2.76 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0221767, upper bound: 0.0221767

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0217452, upper bound: 0.0217332
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0217332, upper bound: 0.0217452
time: 0.67 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.46 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.46
Output dim: 9, lower bound: -0.0217452, upper bound: 0.0217332
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.46
Output dim: 9, lower bound: -0.0217332, upper bound: 0.0217452

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0098324, 0.0098422
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0070150, 0.0070226
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0089983, 0.0090145
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0091899, 0.0091807
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0116240, 0.0116331
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0088374, 0.0088498
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0099249, 0.0099144
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0091456, 0.0091555
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0358821, 0.0358391

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0215796, upper bound: 0.0214368
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0214536, upper bound: 0.0215671
time: 0.71 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0098422, 0.0098324
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0070226, 0.0070150
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0090145, 0.0089983
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0091807, 0.0091899
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0116331, 0.0116240
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0088498, 0.0088374
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0099144, 0.0099249
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0091555, 0.0091456
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0358392, 0.0358821

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0215671, upper bound: 0.0214536
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0214368, upper bound: 0.0215796
time: 0.69 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.64 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.64
Output dim: 9, lower bound: -0.0215796, upper bound: 0.0214368
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.64
Output dim: 9, lower bound: -0.0214536, upper bound: 0.0215671
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.64
Output dim: 9, lower bound: -0.0215671, upper bound: 0.0214536
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.64
Output dim: 9, lower bound: -0.0214368, upper bound: 0.0215796

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0097140, 0.0097522
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0068904, 0.0069192
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0089594, 0.0089789
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0091401, 0.0091233
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0115742, 0.0115920
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0088017, 0.0088182
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0098615, 0.0098387
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0091273, 0.0091389
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0356269, 0.0355474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0192697, upper bound: 0.0201073
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0202191, upper bound: 0.0191532
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0097446, 0.0097238
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0069140, 0.0068981
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0089631, 0.0089755
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0091325, 0.0091319
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0115834, 0.0115832
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0088061, 0.0088140
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0098491, 0.0098519
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0091300, 0.0091372
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0355903, 0.0355903

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0191599, upper bound: 0.0202026
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0201284, upper bound: 0.0192583
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0097238, 0.0097446
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0068981, 0.0069140
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0089755, 0.0089631
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0091319, 0.0091325
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0115832, 0.0115834
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0088140, 0.0088061
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0098519, 0.0098491
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0091372, 0.0091300
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0355903, 0.0355903

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0192583, upper bound: 0.0201284
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0202026, upper bound: 0.0191599
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0097522, 0.0097140
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0069192, 0.0068904
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0089789, 0.0089594
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0091233, 0.0091401
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0115920, 0.0115742
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0088182, 0.0088017
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0098387, 0.0098615
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0091389, 0.0091273
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0355474, 0.0356269

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0191532, upper bound: 0.0202191
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0201073, upper bound: 0.0192697
time: 0.68 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.60 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.60
Output dim: 9, lower bound: -0.0192697, upper bound: 0.0201073
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.60
Output dim: 9, lower bound: -0.0202191, upper bound: 0.0191532
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.60
Output dim: 9, lower bound: -0.0191599, upper bound: 0.0202026
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.60
Output dim: 9, lower bound: -0.0201284, upper bound: 0.0192583
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.60
Output dim: 9, lower bound: -0.0192583, upper bound: 0.0201284
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.60
Output dim: 9, lower bound: -0.0202026, upper bound: 0.0191599
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.60
Output dim: 9, lower bound: -0.0191532, upper bound: 0.0202191
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.60
Output dim: 9, lower bound: -0.0201073, upper bound: 0.0192697

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0088235, 0.0085335
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0062898, 0.0060974
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082697, 0.0079334
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0081796, 0.0085140
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108133, 0.0104216
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0082123, 0.0078968
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0090205, 0.0092970
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084534, 0.0081260
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0309014, 0.0324105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0170447, upper bound: 0.0176465
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0169543, upper bound: 0.0176483
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0084953, 0.0088549
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060686, 0.0063151
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0079140, 0.0082775
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0085254, 0.0081627
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104038, 0.0108256
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078803, 0.0082196
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0093136, 0.0089976
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081144, 0.0084571
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0324626, 0.0308219

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0177885, upper bound: 0.0168234
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0177774, upper bound: 0.0168993
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0088466, 0.0085051
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0063080, 0.0060763
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082729, 0.0079301
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0081719, 0.0085199
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108200, 0.0104128
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0082159, 0.0078927
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0090081, 0.0093071
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084554, 0.0081243
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0308648, 0.0324387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0168993, upper bound: 0.0177774
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0168241, upper bound: 0.0177885
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0085259, 0.0088297
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060922, 0.0062963
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0079177, 0.0082737
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0085184, 0.0081713
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104130, 0.0108181
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078847, 0.0082155
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0093025, 0.0090108
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081172, 0.0084552
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0324291, 0.0308648

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0176483, upper bound: 0.0169520
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0176465, upper bound: 0.0170447
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0088297, 0.0085259
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0062963, 0.0060922
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082737, 0.0079177
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0081713, 0.0085184
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108181, 0.0104130
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0082155, 0.0078847
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0090108, 0.0093025
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084552, 0.0081172
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0308648, 0.0324291

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0170447, upper bound: 0.0176465
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0169520, upper bound: 0.0176483
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0085051, 0.0088466
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060763, 0.0063080
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0079301, 0.0082729
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0085199, 0.0081719
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104128, 0.0108200
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078927, 0.0082159
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0093071, 0.0090081
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081243, 0.0084554
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0324387, 0.0308648

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0177885, upper bound: 0.0168241
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0177774, upper bound: 0.0168993
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0088549, 0.0084953
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0063151, 0.0060686
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082775, 0.0079140
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0081627, 0.0085254
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108256, 0.0104038
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0082196, 0.0078803
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0089976, 0.0093136
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084571, 0.0081144
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0308219, 0.0324626

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0168993, upper bound: 0.0177774
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0168234, upper bound: 0.0177885
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0085335, 0.0088235
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060974, 0.0062898
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0079334, 0.0082697
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0085140, 0.0081796
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104216, 0.0108133
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078968, 0.0082123
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0092970, 0.0090205
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081260, 0.0084534
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0324105, 0.0309014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0176483, upper bound: 0.0169543
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0176465, upper bound: 0.0170447
time: 0.69 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.77 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 9, lower bound: -0.0170447, upper bound: 0.0176465
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 9, lower bound: -0.0169543, upper bound: 0.0176483
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 9, lower bound: -0.0177885, upper bound: 0.0168234
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 9, lower bound: -0.0177774, upper bound: 0.0168993
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 9, lower bound: -0.0168993, upper bound: 0.0177774
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 9, lower bound: -0.0168241, upper bound: 0.0177885
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 9, lower bound: -0.0176483, upper bound: 0.0169520
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 9, lower bound: -0.0176465, upper bound: 0.0170447
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 9, lower bound: -0.0170447, upper bound: 0.0176465
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 9, lower bound: -0.0169520, upper bound: 0.0176483
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 9, lower bound: -0.0177885, upper bound: 0.0168241
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 9, lower bound: -0.0177774, upper bound: 0.0168993
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 9, lower bound: -0.0168993, upper bound: 0.0177774
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 9, lower bound: -0.0168234, upper bound: 0.0177885
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 9, lower bound: -0.0176483, upper bound: 0.0169543
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 9, lower bound: -0.0176465, upper bound: 0.0170447

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0085857, 0.0082203
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0061581, 0.0059145
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0079293, 0.0076057
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0078757, 0.0082169
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104338, 0.0100432
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0079043, 0.0075978
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0087190, 0.0090493
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081012, 0.0078113
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0295343, 0.0310946

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0154192, upper bound: 0.0161193
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0155097, upper bound: 0.0157610
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0088235, 0.0082956
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0062898, 0.0059658
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082697, 0.0075930
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0078825, 0.0085140
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108133, 0.0100420
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0082123, 0.0075889
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0087728, 0.0092970
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084534, 0.0077739
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0295855, 0.0324105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0153391, upper bound: 0.0161195
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0154215, upper bound: 0.0157745
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0082574, 0.0085469
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0059369, 0.0061334
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0075736, 0.0079468
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0082126, 0.0078656
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0100242, 0.0104367
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0075724, 0.0079164
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0090160, 0.0087499
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0077622, 0.0081365
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0310568, 0.0295059

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160426, upper bound: 0.0152686
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162693, upper bound: 0.0150403
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0084953, 0.0086170
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060686, 0.0061834
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0079140, 0.0079371
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0082283, 0.0081627
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104038, 0.0104461
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078803, 0.0079116
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0090658, 0.0089976
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081144, 0.0081050
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0311467, 0.0308219

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160396, upper bound: 0.0153344
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162662, upper bound: 0.0151303
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0086087, 0.0081933
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0061763, 0.0058935
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0079325, 0.0076003
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0078664, 0.0082228
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104405, 0.0100330
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0079080, 0.0075920
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0087036, 0.0090593
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081032, 0.0078080
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0294894, 0.0311227

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0151303, upper bound: 0.0162662
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0153344, upper bound: 0.0160396
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0088466, 0.0082673
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0063080, 0.0059446
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082729, 0.0075897
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0078748, 0.0085199
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108200, 0.0100333
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0082159, 0.0075847
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0087604, 0.0093071
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084554, 0.0077721
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0295489, 0.0324387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0150420, upper bound: 0.0162693
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0152686, upper bound: 0.0160439
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0082880, 0.0085208
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0059605, 0.0061146
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0075773, 0.0079403
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0082040, 0.0078742
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0100334, 0.0104269
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0075768, 0.0079098
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0090037, 0.0087631
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0077650, 0.0081326
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0310194, 0.0295488

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0157741, upper bound: 0.0154212
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0161195, upper bound: 0.0153391
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0085259, 0.0085919
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060922, 0.0061646
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0079177, 0.0079333
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0082213, 0.0081713
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104130, 0.0104386
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078847, 0.0079075
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0090548, 0.0090108
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081172, 0.0081031
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0311132, 0.0308648

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0157610, upper bound: 0.0155097
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0161193, upper bound: 0.0154192
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0085919, 0.0082198
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0061646, 0.0059127
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0079333, 0.0076075
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0078781, 0.0082213
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104386, 0.0100461
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0079075, 0.0075992
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0087192, 0.0090548
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081031, 0.0078126
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0295462, 0.0311132

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0154192, upper bound: 0.0161193
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0155097, upper bound: 0.0157610
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0085208, 0.0082880
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0061146, 0.0059605
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0079403, 0.0075773
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0078742, 0.0082040
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104269, 0.0100334
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0079098, 0.0075768
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0087631, 0.0090037
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081326, 0.0077650
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0295488, 0.0310194

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0153391, upper bound: 0.0161195
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0154212, upper bound: 0.0157741
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0082673, 0.0085475
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0059446, 0.0061338
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0075897, 0.0079483
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0082145, 0.0078748
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0100333, 0.0104387
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0075847, 0.0079181
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0090170, 0.0087604
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0077721, 0.0081379
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0310645, 0.0295489

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160439, upper bound: 0.0152686
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162693, upper bound: 0.0150420
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0081933, 0.0086087
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0058935, 0.0061763
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0076003, 0.0079325
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0082228, 0.0078664
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0100330, 0.0104405
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0075920, 0.0079080
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0090593, 0.0087036
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0078080, 0.0081032
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0311227, 0.0294894

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160396, upper bound: 0.0153344
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162662, upper bound: 0.0151303
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0086170, 0.0081921
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0061834, 0.0058913
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0079371, 0.0076019
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0078686, 0.0082283
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104461, 0.0100353
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0079116, 0.0075933
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0087037, 0.0090658
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081050, 0.0078090
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0294999, 0.0311467

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0151303, upper bound: 0.0162662
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0153344, upper bound: 0.0160396
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0085469, 0.0082574
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0061334, 0.0059369
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0079468, 0.0075736
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0078656, 0.0082126
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104367, 0.0100242
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0079164, 0.0075724
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0087499, 0.0090160
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081365, 0.0077622
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0295059, 0.0310568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0150403, upper bound: 0.0162693
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0152686, upper bound: 0.0160426
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0082956, 0.0085218
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0059658, 0.0061144
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0075930, 0.0079422
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0082054, 0.0078825
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0100420, 0.0104291
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0075889, 0.0079116
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0090042, 0.0087728
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0077739, 0.0081342
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0310245, 0.0295855

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0157745, upper bound: 0.0154215
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0161195, upper bound: 0.0153391
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0082203, 0.0085857
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0059145, 0.0061581
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0076057, 0.0079293
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0082169, 0.0078757
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0100432, 0.0104338
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0075978, 0.0079043
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0090493, 0.0087190
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0078113, 0.0081012
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0310946, 0.0295343

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0157610, upper bound: 0.0155097
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0161193, upper bound: 0.0154192
time: 0.70 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.68 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 9, lower bound: -0.0154192, upper bound: 0.0161193
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 9, lower bound: -0.0155097, upper bound: 0.0157610
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 9, lower bound: -0.0153391, upper bound: 0.0161195
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 9, lower bound: -0.0154215, upper bound: 0.0157745
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 9, lower bound: -0.0160426, upper bound: 0.0152686
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 9, lower bound: -0.0162693, upper bound: 0.0150403
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 9, lower bound: -0.0160396, upper bound: 0.0153344
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 9, lower bound: -0.0162662, upper bound: 0.0151303
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 9, lower bound: -0.0151303, upper bound: 0.0162662
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 9, lower bound: -0.0153344, upper bound: 0.0160396
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 9, lower bound: -0.0150420, upper bound: 0.0162693
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 9, lower bound: -0.0152686, upper bound: 0.0160439
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 9, lower bound: -0.0157741, upper bound: 0.0154212
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 9, lower bound: -0.0161195, upper bound: 0.0153391
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 9, lower bound: -0.0157610, upper bound: 0.0155097
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 9, lower bound: -0.0161193, upper bound: 0.0154192
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 9, lower bound: -0.0154192, upper bound: 0.0161193
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 9, lower bound: -0.0155097, upper bound: 0.0157610
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 9, lower bound: -0.0153391, upper bound: 0.0161195
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 9, lower bound: -0.0154212, upper bound: 0.0157741
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 9, lower bound: -0.0160439, upper bound: 0.0152686
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 9, lower bound: -0.0162693, upper bound: 0.0150420
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 9, lower bound: -0.0160396, upper bound: 0.0153344
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 9, lower bound: -0.0162662, upper bound: 0.0151303
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 9, lower bound: -0.0151303, upper bound: 0.0162662
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 9, lower bound: -0.0153344, upper bound: 0.0160396
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 9, lower bound: -0.0150403, upper bound: 0.0162693
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 9, lower bound: -0.0152686, upper bound: 0.0160426
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 9, lower bound: -0.0157745, upper bound: 0.0154215
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 9, lower bound: -0.0161195, upper bound: 0.0153391
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 9, lower bound: -0.0157610, upper bound: 0.0155097
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 9, lower bound: -0.0161193, upper bound: 0.0154192

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0087043, 0.0083757
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0061561, 0.0059390
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0073545, 0.0068878
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0071831, 0.0076045
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0098745, 0.0093803
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0072683, 0.0068471
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0081811, 0.0085220
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0078039, 0.0073567
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0273629, 0.0292595

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135212, upper bound: 0.0140975
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135212, upper bound: 0.0140975
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0086516, 0.0084142
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0061213, 0.0059637
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0072063, 0.0070182
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0072700, 0.0074922
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0097465, 0.0094828
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0071439, 0.0069529
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0082454, 0.0084314
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0076686, 0.0074766
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0277503, 0.0287541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136363, upper bound: 0.0137828
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136363, upper bound: 0.0137828
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0087043, 0.0083757
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0061561, 0.0059390
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0073545, 0.0068878
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0071831, 0.0076045
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0098745, 0.0093803
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0072683, 0.0068471
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0081811, 0.0085220
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0078039, 0.0073567
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0273629, 0.0292595

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135211, upper bound: 0.0140975
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135211, upper bound: 0.0140975
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0086516, 0.0084142
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0061213, 0.0059637
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0072063, 0.0070182
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0072700, 0.0074922
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0097465, 0.0094828
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0071439, 0.0069529
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0082454, 0.0084314
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0076686, 0.0074766
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0277503, 0.0287541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136337, upper bound: 0.0137828
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136337, upper bound: 0.0137828
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0083760, 0.0086892
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0059349, 0.0061513
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0069988, 0.0072206
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0075088, 0.0072532
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0094650, 0.0097639
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0069364, 0.0071559
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0084525, 0.0082226
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0074649, 0.0076799
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0288306, 0.0276708

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140420, upper bound: 0.0135248
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140420, upper bound: 0.0135248
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0083288, 0.0087356
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0059037, 0.0061813
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0068641, 0.0073623
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0076158, 0.0071536
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0093509, 0.0098869
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0068213, 0.0072756
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0085385, 0.0081448
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0073380, 0.0078077
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0293116, 0.0272253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0142283, upper bound: 0.0133052
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0142283, upper bound: 0.0133052
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0083760, 0.0086892
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0059349, 0.0061513
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0069988, 0.0072206
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0075088, 0.0072532
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0094650, 0.0097639
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0069364, 0.0071559
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0084525, 0.0082226
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0074649, 0.0076799
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0288306, 0.0276708

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140420, upper bound: 0.0135267
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140420, upper bound: 0.0135267
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0083288, 0.0087356
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0059037, 0.0061813
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0068641, 0.0073623
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0076158, 0.0071536
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0093509, 0.0098869
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0068213, 0.0072756
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0085385, 0.0081448
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0073380, 0.0078077
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0293116, 0.0272253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0142283, upper bound: 0.0133053
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0142283, upper bound: 0.0133053
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0087273, 0.0083364
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0061742, 0.0059105
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0073577, 0.0068683
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0071582, 0.0076103
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0098813, 0.0093532
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0072720, 0.0068261
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0081521, 0.0085320
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0078060, 0.0073403
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0272484, 0.0292876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133053, upper bound: 0.0142283
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133053, upper bound: 0.0142283
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0086841, 0.0083859
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0061458, 0.0059425
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0072226, 0.0070149
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0072624, 0.0075117
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0097682, 0.0094741
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0071598, 0.0069487
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0082330, 0.0084511
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0076841, 0.0074748
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0277138, 0.0288435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135267, upper bound: 0.0140420
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135267, upper bound: 0.0140420
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0087273, 0.0083364
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0061742, 0.0059105
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0073577, 0.0068683
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0071582, 0.0076103
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0098813, 0.0093532
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0072720, 0.0068261
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0081521, 0.0085320
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0078060, 0.0073403
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0272484, 0.0292876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133052, upper bound: 0.0142283
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133052, upper bound: 0.0142283
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0086841, 0.0083859
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0061458, 0.0059425
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0072226, 0.0070149
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0072624, 0.0075117
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0097682, 0.0094741
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0071598, 0.0069487
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0082330, 0.0084511
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0076841, 0.0074748
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0277138, 0.0288435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135248, upper bound: 0.0140420
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135248, upper bound: 0.0140420
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0084066, 0.0086548
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0059584, 0.0061261
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0070025, 0.0072049
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0074913, 0.0072618
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0094742, 0.0097442
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0069408, 0.0071407
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0084339, 0.0082358
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0074677, 0.0076639
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0287518, 0.0277137

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137828, upper bound: 0.0136337
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137828, upper bound: 0.0136337
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0083731, 0.0087105
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0059361, 0.0061625
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0068866, 0.0073585
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0076088, 0.0071791
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0093783, 0.0098794
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0068441, 0.0072715
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0085274, 0.0081736
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0073585, 0.0078058
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0292781, 0.0273419

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140975, upper bound: 0.0135211
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140975, upper bound: 0.0135211
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0084066, 0.0086548
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0059584, 0.0061261
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0070025, 0.0072049
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0074913, 0.0072618
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0094742, 0.0097442
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0069408, 0.0071407
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0084339, 0.0082358
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0074677, 0.0076639
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0287518, 0.0277137

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137828, upper bound: 0.0136363
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137828, upper bound: 0.0136363
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0083731, 0.0087105
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0059361, 0.0061625
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0068866, 0.0073585
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0076088, 0.0071791
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0093783, 0.0098794
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0068441, 0.0072715
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0085274, 0.0081736
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0073585, 0.0078058
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0292781, 0.0273419

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140975, upper bound: 0.0135212
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140975, upper bound: 0.0135212
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0087105, 0.0083731
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0061625, 0.0059361
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0073585, 0.0068866
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0071791, 0.0076088
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0098794, 0.0093783
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0072715, 0.0068441
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0081736, 0.0085274
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0078058, 0.0073585
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0273419, 0.0292781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135212, upper bound: 0.0140975
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135212, upper bound: 0.0140975
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0086548, 0.0084066
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0061261, 0.0059584
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0072049, 0.0070025
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0072618, 0.0074913
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0097442, 0.0094742
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0071407, 0.0069408
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0082358, 0.0084339
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0076639, 0.0074677
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0277137, 0.0287518

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136363, upper bound: 0.0137828
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136363, upper bound: 0.0137828
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0087105, 0.0083731
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0061625, 0.0059361
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0073585, 0.0068866
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0071791, 0.0076088
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0098794, 0.0093783
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0072715, 0.0068441
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0081736, 0.0085274
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0078058, 0.0073585
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0273419, 0.0292781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135211, upper bound: 0.0140975
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135211, upper bound: 0.0140975
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0086548, 0.0084066
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0061261, 0.0059584
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0072049, 0.0070025
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0072618, 0.0074913
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0097442, 0.0094742
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0071407, 0.0069408
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0082358, 0.0084339
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0076639, 0.0074677
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0277137, 0.0287518

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136337, upper bound: 0.0137828
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136337, upper bound: 0.0137828
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0083859, 0.0086841
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0059425, 0.0061458
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0070149, 0.0072226
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0075117, 0.0072624
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0094741, 0.0097682
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0069487, 0.0071598
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0084511, 0.0082330
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0074748, 0.0076841
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0288435, 0.0277138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140420, upper bound: 0.0135248
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140420, upper bound: 0.0135248
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0083364, 0.0087273
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0059105, 0.0061742
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0068683, 0.0073577
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0076103, 0.0071582
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0093532, 0.0098813
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0068261, 0.0072720
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0085320, 0.0081521
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0073403, 0.0078060
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0292876, 0.0272484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0142283, upper bound: 0.0133052
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0142283, upper bound: 0.0133052
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0083859, 0.0086841
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0059425, 0.0061458
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0070149, 0.0072226
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0075117, 0.0072624
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0094741, 0.0097682
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0069487, 0.0071598
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0084511, 0.0082330
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0074748, 0.0076841
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0288435, 0.0277138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140420, upper bound: 0.0135267
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140420, upper bound: 0.0135267
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0083364, 0.0087273
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0059105, 0.0061742
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0068683, 0.0073577
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0076103, 0.0071582
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0093532, 0.0098813
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0068261, 0.0072720
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0085320, 0.0081521
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0073403, 0.0078060
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0292876, 0.0272484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0142283, upper bound: 0.0133053
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0142283, upper bound: 0.0133053
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0087356, 0.0083288
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0061813, 0.0059037
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0073623, 0.0068641
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0071536, 0.0076158
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0098869, 0.0093509
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0072756, 0.0068213
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0081448, 0.0085385
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0078077, 0.0073380
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0272253, 0.0293116

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133053, upper bound: 0.0142283
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133053, upper bound: 0.0142283
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0086892, 0.0083760
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0061513, 0.0059349
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0072206, 0.0069988
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0072532, 0.0075088
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0097639, 0.0094650
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0071559, 0.0069364
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0082226, 0.0084525
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0076799, 0.0074649
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0276708, 0.0288306

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135267, upper bound: 0.0140420
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135267, upper bound: 0.0140420
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0087356, 0.0083288
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0061813, 0.0059037
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0073623, 0.0068641
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0071536, 0.0076158
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0098869, 0.0093509
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0072756, 0.0068213
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0081448, 0.0085385
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0078077, 0.0073380
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0272253, 0.0293116

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133052, upper bound: 0.0142283
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133052, upper bound: 0.0142283
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0086892, 0.0083760
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0061513, 0.0059349
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0072206, 0.0069988
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0072532, 0.0075088
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0097639, 0.0094650
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0071559, 0.0069364
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0082226, 0.0084525
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0076799, 0.0074649
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0276708, 0.0288306

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135248, upper bound: 0.0140420
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135248, upper bound: 0.0140420
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0084142, 0.0086516
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0059637, 0.0061213
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0070182, 0.0072063
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0074922, 0.0072700
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0094828, 0.0097465
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0069529, 0.0071439
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0084314, 0.0082454
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0074766, 0.0076686
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0287541, 0.0277504

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137828, upper bound: 0.0136337
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137828, upper bound: 0.0136337
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0083757, 0.0087043
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0059390, 0.0061561
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0068878, 0.0073545
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0076045, 0.0071831
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0093803, 0.0098745
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0068471, 0.0072683
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0085220, 0.0081811
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0073567, 0.0078039
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0292595, 0.0273630

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140975, upper bound: 0.0135211
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140975, upper bound: 0.0135211
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0084142, 0.0086516
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0059637, 0.0061213
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0070182, 0.0072063
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0074922, 0.0072700
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0094828, 0.0097465
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0069529, 0.0071439
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0084314, 0.0082454
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0074766, 0.0076686
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0287541, 0.0277504

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137828, upper bound: 0.0136363
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137828, upper bound: 0.0136363
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0083757, 0.0087043
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0059390, 0.0061561
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0068878, 0.0073545
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0076045, 0.0071831
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0093803, 0.0098745
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0068471, 0.0072683
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0085220, 0.0081811
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0073567, 0.0078039
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0292595, 0.0273630

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140975, upper bound: 0.0135212
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140975, upper bound: 0.0135212
time: 0.69 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.75 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0135212, upper bound: 0.0140975
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0135212, upper bound: 0.0140975
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0136363, upper bound: 0.0137828
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0136363, upper bound: 0.0137828
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0135211, upper bound: 0.0140975
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0135211, upper bound: 0.0140975
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0136337, upper bound: 0.0137828
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0136337, upper bound: 0.0137828
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0140420, upper bound: 0.0135248
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0140420, upper bound: 0.0135248
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0142283, upper bound: 0.0133052
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0142283, upper bound: 0.0133052
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0140420, upper bound: 0.0135267
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0140420, upper bound: 0.0135267
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0142283, upper bound: 0.0133053
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0142283, upper bound: 0.0133053
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0133053, upper bound: 0.0142283
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0133053, upper bound: 0.0142283
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0135267, upper bound: 0.0140420
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0135267, upper bound: 0.0140420
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0133052, upper bound: 0.0142283
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0133052, upper bound: 0.0142283
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0135248, upper bound: 0.0140420
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0135248, upper bound: 0.0140420
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0137828, upper bound: 0.0136337
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0137828, upper bound: 0.0136337
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0140975, upper bound: 0.0135211
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0140975, upper bound: 0.0135211
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0137828, upper bound: 0.0136363
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0137828, upper bound: 0.0136363
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0140975, upper bound: 0.0135212
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0140975, upper bound: 0.0135212
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0135212, upper bound: 0.0140975
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0135212, upper bound: 0.0140975
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0136363, upper bound: 0.0137828
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0136363, upper bound: 0.0137828
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0135211, upper bound: 0.0140975
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0135211, upper bound: 0.0140975
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0136337, upper bound: 0.0137828
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0136337, upper bound: 0.0137828
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0140420, upper bound: 0.0135248
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0140420, upper bound: 0.0135248
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0142283, upper bound: 0.0133052
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0142283, upper bound: 0.0133052
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0140420, upper bound: 0.0135267
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0140420, upper bound: 0.0135267
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0142283, upper bound: 0.0133053
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0142283, upper bound: 0.0133053
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0133053, upper bound: 0.0142283
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0133053, upper bound: 0.0142283
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0135267, upper bound: 0.0140420
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0135267, upper bound: 0.0140420
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0133052, upper bound: 0.0142283
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0133052, upper bound: 0.0142283
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0135248, upper bound: 0.0140420
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0135248, upper bound: 0.0140420
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0137828, upper bound: 0.0136337
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0137828, upper bound: 0.0136337
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0140975, upper bound: 0.0135211
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0140975, upper bound: 0.0135211
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0137828, upper bound: 0.0136363
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0137828, upper bound: 0.0136363
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0140975, upper bound: 0.0135212
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 9, lower bound: -0.0140975, upper bound: 0.0135212

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0087988, 0.0085042
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0062727, 0.0060769
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082462, 0.0078976
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0081662, 0.0085056
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108061, 0.0104101
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0081953, 0.0078705
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0090111, 0.0092908
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084363, 0.0080974
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0308405, 0.0323750

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127039, upper bound: 0.0132780
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127014, upper bound: 0.0131593
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0087943, 0.0085335
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0062693, 0.0060974
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082697, 0.0079099
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0081712, 0.0085140
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108133, 0.0104144
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0082123, 0.0078798
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0090142, 0.0092970
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084534, 0.0081090
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0308658, 0.0324105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127039, upper bound: 0.0132780
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127014, upper bound: 0.0131593
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0087988, 0.0085042
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0062727, 0.0060769
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082462, 0.0078976
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0081662, 0.0085056
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108061, 0.0104101
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0081953, 0.0078705
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0090111, 0.0092908
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084363, 0.0080974
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0308405, 0.0323750

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128298, upper bound: 0.0129523
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128244, upper bound: 0.0128579
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0087943, 0.0085335
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0062693, 0.0060974
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082697, 0.0079099
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0081712, 0.0085140
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108133, 0.0104144
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0082123, 0.0078798
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0090142, 0.0092970
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084534, 0.0081090
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0308658, 0.0324105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128298, upper bound: 0.0129523
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128244, upper bound: 0.0128579
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0087988, 0.0085042
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0062727, 0.0060769
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082462, 0.0078976
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0081662, 0.0085056
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108061, 0.0104101
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0081953, 0.0078705
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0090111, 0.0092908
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084363, 0.0080974
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0308405, 0.0323750

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127008, upper bound: 0.0132780
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126994, upper bound: 0.0131742
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0087943, 0.0085335
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0062693, 0.0060974
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082697, 0.0079099
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0081712, 0.0085140
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108133, 0.0104144
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0082123, 0.0078798
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0090142, 0.0092970
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084534, 0.0081090
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0308658, 0.0324105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127008, upper bound: 0.0132780
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126994, upper bound: 0.0131742
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0087988, 0.0085042
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0062727, 0.0060769
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082462, 0.0078976
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0081662, 0.0085056
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108061, 0.0104101
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0081953, 0.0078705
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0090111, 0.0092908
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084363, 0.0080974
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0308405, 0.0323750

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128219, upper bound: 0.0129523
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128182, upper bound: 0.0128610
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0087943, 0.0085335
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0062693, 0.0060974
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082697, 0.0079099
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0081712, 0.0085140
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108133, 0.0104144
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0082123, 0.0078798
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0090142, 0.0092970
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084534, 0.0081090
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0308658, 0.0324105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128219, upper bound: 0.0129523
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128182, upper bound: 0.0128610
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0084847, 0.0088256
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060604, 0.0062945
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0078904, 0.0082572
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0085167, 0.0081543
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0103966, 0.0108183
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078633, 0.0082038
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0093058, 0.0089913
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0080974, 0.0084429
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0324257, 0.0307863

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131641, upper bound: 0.0127156
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132196, upper bound: 0.0127165
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0084660, 0.0088549
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060481, 0.0063151
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0079140, 0.0082540
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0085169, 0.0081627
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104038, 0.0108185
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078803, 0.0082026
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0093073, 0.0089976
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081144, 0.0084401
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0324270, 0.0308219

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131641, upper bound: 0.0127156
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132196, upper bound: 0.0127165
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0084847, 0.0088256
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060604, 0.0062945
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0078904, 0.0082572
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0085167, 0.0081543
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0103966, 0.0108183
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078633, 0.0082038
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0093058, 0.0089913
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0080974, 0.0084429
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0324257, 0.0307863

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133211, upper bound: 0.0124817
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0134081, upper bound: 0.0124780
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0084660, 0.0088549
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060481, 0.0063151
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0079140, 0.0082540
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0085169, 0.0081627
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104038, 0.0108185
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078803, 0.0082026
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0093073, 0.0089976
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081144, 0.0084401
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0324270, 0.0308219

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133211, upper bound: 0.0124817
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0134081, upper bound: 0.0124780
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0084847, 0.0088256
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060604, 0.0062945
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0078904, 0.0082572
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0085167, 0.0081543
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0103966, 0.0108183
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078633, 0.0082038
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0093058, 0.0089913
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0080974, 0.0084429
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0324257, 0.0307863

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131628, upper bound: 0.0127206
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132196, upper bound: 0.0127246
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0084660, 0.0088549
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060481, 0.0063151
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0079140, 0.0082540
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0085169, 0.0081627
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104038, 0.0108185
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078803, 0.0082026
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0093073, 0.0089976
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081144, 0.0084401
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0324270, 0.0308219

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131628, upper bound: 0.0127206
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132196, upper bound: 0.0127246
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0084847, 0.0088256
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060604, 0.0062945
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0078904, 0.0082572
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0085167, 0.0081543
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0103966, 0.0108183
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078633, 0.0082038
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0093058, 0.0089913
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0080974, 0.0084429
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0324257, 0.0307863

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133185, upper bound: 0.0124840
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0134081, upper bound: 0.0124820
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0084660, 0.0088549
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060481, 0.0063151
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0079140, 0.0082540
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0085169, 0.0081627
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104038, 0.0108185
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078803, 0.0082026
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0093073, 0.0089976
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081144, 0.0084401
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0324270, 0.0308219

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133185, upper bound: 0.0124840
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0134081, upper bound: 0.0124820
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0088241, 0.0084759
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0062920, 0.0060557
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082493, 0.0078956
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0081581, 0.0085115
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108129, 0.0104013
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0081989, 0.0078678
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0089988, 0.0093008
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084384, 0.0080963
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0308030, 0.0324031

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0124820, upper bound: 0.0134081
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0124840, upper bound: 0.0133185
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0088173, 0.0085051
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0062874, 0.0060763
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082729, 0.0079066
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0081635, 0.0085199
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108200, 0.0104057
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0082159, 0.0078757
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0090018, 0.0093071
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084554, 0.0081073
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0308292, 0.0324387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0124820, upper bound: 0.0134081
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0124840, upper bound: 0.0133185
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0088241, 0.0084759
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0062920, 0.0060557
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082493, 0.0078956
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0081581, 0.0085115
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108129, 0.0104013
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0081989, 0.0078678
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0089988, 0.0093008
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084384, 0.0080963
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0308030, 0.0324031

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127246, upper bound: 0.0132196
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127206, upper bound: 0.0131628
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0088173, 0.0085051
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0062874, 0.0060763
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082729, 0.0079066
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0081635, 0.0085199
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108200, 0.0104057
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0082159, 0.0078757
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0090018, 0.0093071
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084554, 0.0081073
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0308292, 0.0324387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127246, upper bound: 0.0132196
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127206, upper bound: 0.0131628
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0088241, 0.0084759
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0062920, 0.0060557
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082493, 0.0078956
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0081581, 0.0085115
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108129, 0.0104013
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0081989, 0.0078678
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0089988, 0.0093008
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084384, 0.0080963
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0308030, 0.0324031

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0124780, upper bound: 0.0134081
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0124817, upper bound: 0.0133211
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0088173, 0.0085051
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0062874, 0.0060763
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082729, 0.0079066
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0081635, 0.0085199
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108200, 0.0104057
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0082159, 0.0078757
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0090018, 0.0093071
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084554, 0.0081073
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0308292, 0.0324387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0124780, upper bound: 0.0134081
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0124817, upper bound: 0.0133211
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0088241, 0.0084759
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0062920, 0.0060557
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082493, 0.0078956
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0081581, 0.0085115
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108129, 0.0104013
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0081989, 0.0078678
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0089988, 0.0093008
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084384, 0.0080963
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0308030, 0.0324031

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127165, upper bound: 0.0132196
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127156, upper bound: 0.0131641
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0088173, 0.0085051
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0062874, 0.0060763
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082729, 0.0079066
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0081635, 0.0085199
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108200, 0.0104057
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0082159, 0.0078757
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0090018, 0.0093071
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084554, 0.0081073
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0308292, 0.0324387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127165, upper bound: 0.0132196
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127156, upper bound: 0.0131641
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0085172, 0.0088005
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060859, 0.0062757
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0078942, 0.0082550
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0085100, 0.0081629
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104058, 0.0108111
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078677, 0.0082004
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0092952, 0.0090046
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081002, 0.0084415
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0323941, 0.0308292

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128610, upper bound: 0.0128182
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129523, upper bound: 0.0128219
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0084966, 0.0088297
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060716, 0.0062963
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0079177, 0.0082501
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0085099, 0.0081713
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104130, 0.0108110
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078847, 0.0081985
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0092962, 0.0090108
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081172, 0.0084382
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0323935, 0.0308648

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128610, upper bound: 0.0128182
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129523, upper bound: 0.0128219
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0085172, 0.0088005
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060859, 0.0062757
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0078942, 0.0082550
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0085100, 0.0081629
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104058, 0.0108111
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078677, 0.0082004
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0092952, 0.0090046
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081002, 0.0084415
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0323941, 0.0308292

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131745, upper bound: 0.0126994
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132780, upper bound: 0.0127008
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0084966, 0.0088297
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060716, 0.0062963
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0079177, 0.0082501
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0085099, 0.0081713
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104130, 0.0108110
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078847, 0.0081985
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0092962, 0.0090108
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081172, 0.0084382
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0323935, 0.0308648

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131745, upper bound: 0.0126994
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132780, upper bound: 0.0127008
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0085172, 0.0088005
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060859, 0.0062757
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0078942, 0.0082550
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0085100, 0.0081629
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104058, 0.0108111
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078677, 0.0082004
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0092952, 0.0090046
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081002, 0.0084415
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0323941, 0.0308292

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128579, upper bound: 0.0128244
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129523, upper bound: 0.0128298
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0084966, 0.0088297
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060716, 0.0062963
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0079177, 0.0082501
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0085099, 0.0081713
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104130, 0.0108110
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078847, 0.0081985
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0092962, 0.0090108
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081172, 0.0084382
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0323935, 0.0308648

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128579, upper bound: 0.0128244
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129523, upper bound: 0.0128298
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0085172, 0.0088005
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060859, 0.0062757
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0078942, 0.0082550
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0085100, 0.0081629
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104058, 0.0108111
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078677, 0.0082004
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0092952, 0.0090046
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081002, 0.0084415
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0323941, 0.0308292

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131642, upper bound: 0.0127014
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132780, upper bound: 0.0127039
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0084966, 0.0088297
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060716, 0.0062963
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0079177, 0.0082501
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0085099, 0.0081713
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104130, 0.0108110
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078847, 0.0081985
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0092962, 0.0090108
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081172, 0.0084382
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0323935, 0.0308648

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131642, upper bound: 0.0127014
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132780, upper bound: 0.0127039
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0088090, 0.0084966
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0062819, 0.0060716
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082501, 0.0078794
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0081573, 0.0085099
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108110, 0.0104013
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0081985, 0.0078581
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0090006, 0.0092962
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084382, 0.0080880
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0308026, 0.0323935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127039, upper bound: 0.0132780
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127014, upper bound: 0.0131642
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0088005, 0.0085259
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0062757, 0.0060922
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082737, 0.0078942
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0081629, 0.0085184
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108181, 0.0104058
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0082155, 0.0078677
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0090046, 0.0093025
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084552, 0.0081002
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0308292, 0.0324291

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127039, upper bound: 0.0132780
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127014, upper bound: 0.0131642
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0088090, 0.0084966
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0062819, 0.0060716
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082501, 0.0078794
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0081573, 0.0085099
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108110, 0.0104013
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0081985, 0.0078581
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0090006, 0.0092962
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084382, 0.0080880
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0308026, 0.0323935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128298, upper bound: 0.0129523
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128244, upper bound: 0.0128579
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0088005, 0.0085259
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0062757, 0.0060922
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082737, 0.0078942
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0081629, 0.0085184
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108181, 0.0104058
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0082155, 0.0078677
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0090046, 0.0093025
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084552, 0.0081002
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0308292, 0.0324291

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128298, upper bound: 0.0129523
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128244, upper bound: 0.0128579
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0088090, 0.0084966
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0062819, 0.0060716
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082501, 0.0078794
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0081573, 0.0085099
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108110, 0.0104013
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0081985, 0.0078581
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0090006, 0.0092962
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084382, 0.0080880
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0308026, 0.0323935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127008, upper bound: 0.0132780
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126994, upper bound: 0.0131745
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0088005, 0.0085259
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0062757, 0.0060922
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082737, 0.0078942
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0081629, 0.0085184
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108181, 0.0104058
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0082155, 0.0078677
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0090046, 0.0093025
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084552, 0.0081002
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0308292, 0.0324291

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127008, upper bound: 0.0132780
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126994, upper bound: 0.0131745
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0088090, 0.0084966
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0062819, 0.0060716
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082501, 0.0078794
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0081573, 0.0085099
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108110, 0.0104013
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0081985, 0.0078581
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0090006, 0.0092962
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084382, 0.0080880
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0308026, 0.0323935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128219, upper bound: 0.0129523
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128182, upper bound: 0.0128610
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0088005, 0.0085259
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0062757, 0.0060922
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082737, 0.0078942
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0081629, 0.0085184
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108181, 0.0104058
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0082155, 0.0078677
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0090046, 0.0093025
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084552, 0.0081002
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0308292, 0.0324291

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128219, upper bound: 0.0129523
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128182, upper bound: 0.0128610
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0084933, 0.0088173
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060672, 0.0062874
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0079066, 0.0082531
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0085112, 0.0081635
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104057, 0.0108130
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078757, 0.0082005
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0092992, 0.0090018
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081073, 0.0084408
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0323995, 0.0308292

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131641, upper bound: 0.0127156
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132196, upper bound: 0.0127165
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0084759, 0.0088466
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060557, 0.0063080
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0079301, 0.0082493
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0085115, 0.0081719
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104128, 0.0108129
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078927, 0.0081989
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0093008, 0.0090081
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081243, 0.0084384
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0324031, 0.0308648

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131641, upper bound: 0.0127156
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132196, upper bound: 0.0127165
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0084933, 0.0088173
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060672, 0.0062874
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0079066, 0.0082531
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0085112, 0.0081635
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104057, 0.0108130
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078757, 0.0082005
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0092992, 0.0090018
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081073, 0.0084408
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0323995, 0.0308292

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133211, upper bound: 0.0124817
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0134081, upper bound: 0.0124780
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0084759, 0.0088466
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060557, 0.0063080
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0079301, 0.0082493
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0085115, 0.0081719
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104128, 0.0108129
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078927, 0.0081989
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0093008, 0.0090081
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081243, 0.0084384
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0324031, 0.0308648

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133211, upper bound: 0.0124817
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0134081, upper bound: 0.0124780
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0084933, 0.0088173
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060672, 0.0062874
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0079066, 0.0082531
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0085112, 0.0081635
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104057, 0.0108130
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078757, 0.0082005
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0092992, 0.0090018
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081073, 0.0084408
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0323995, 0.0308292

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131628, upper bound: 0.0127206
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132196, upper bound: 0.0127246
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0084759, 0.0088466
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060557, 0.0063080
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0079301, 0.0082493
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0085115, 0.0081719
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104128, 0.0108129
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078927, 0.0081989
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0093008, 0.0090081
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081243, 0.0084384
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0324031, 0.0308648

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131628, upper bound: 0.0127206
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132196, upper bound: 0.0127246
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0084933, 0.0088173
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060672, 0.0062874
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0079066, 0.0082531
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0085112, 0.0081635
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104057, 0.0108130
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078757, 0.0082005
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0092992, 0.0090018
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081073, 0.0084408
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0323995, 0.0308292

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133185, upper bound: 0.0124840
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0134081, upper bound: 0.0124820
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0084759, 0.0088466
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060557, 0.0063080
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0079301, 0.0082493
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0085115, 0.0081719
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104128, 0.0108129
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078927, 0.0081989
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0093008, 0.0090081
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081243, 0.0084384
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0324031, 0.0308648

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133185, upper bound: 0.0124840
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0134081, upper bound: 0.0124820
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0088350, 0.0084660
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0063007, 0.0060481
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082540, 0.0078774
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0081491, 0.0085169
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108185, 0.0103925
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0082026, 0.0078546
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0089884, 0.0093073
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084401, 0.0080867
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0307619, 0.0324270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0124820, upper bound: 0.0134081
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0124840, upper bound: 0.0133185
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0088256, 0.0084953
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0062945, 0.0060686
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082775, 0.0078904
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0081543, 0.0085254
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108256, 0.0103966
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0082196, 0.0078633
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0089913, 0.0093136
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084571, 0.0080974
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0307863, 0.0324626

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0124820, upper bound: 0.0134081
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0124840, upper bound: 0.0133185
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0088350, 0.0084660
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0063007, 0.0060481
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082540, 0.0078774
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0081491, 0.0085169
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108185, 0.0103925
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0082026, 0.0078546
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0089884, 0.0093073
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084401, 0.0080867
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0307619, 0.0324270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127246, upper bound: 0.0132196
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127206, upper bound: 0.0131628
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0088256, 0.0084953
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0062945, 0.0060686
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082775, 0.0078904
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0081543, 0.0085254
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108256, 0.0103966
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0082196, 0.0078633
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0089913, 0.0093136
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084571, 0.0080974
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0307863, 0.0324626

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127246, upper bound: 0.0132196
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127206, upper bound: 0.0131628
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0088350, 0.0084660
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0063007, 0.0060481
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082540, 0.0078774
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0081491, 0.0085169
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108185, 0.0103925
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0082026, 0.0078546
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0089884, 0.0093073
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084401, 0.0080867
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0307619, 0.0324270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0124780, upper bound: 0.0134081
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0124817, upper bound: 0.0133211
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0088256, 0.0084953
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0062945, 0.0060686
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082775, 0.0078904
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0081543, 0.0085254
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108256, 0.0103966
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0082196, 0.0078633
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0089913, 0.0093136
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084571, 0.0080974
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0307863, 0.0324626

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0124780, upper bound: 0.0134081
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0124817, upper bound: 0.0133211
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0088350, 0.0084660
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0063007, 0.0060481
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082540, 0.0078774
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0081491, 0.0085169
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108185, 0.0103925
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0082026, 0.0078546
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0089884, 0.0093073
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084401, 0.0080867
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0307619, 0.0324270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127165, upper bound: 0.0132196
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127156, upper bound: 0.0131641
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0088256, 0.0084953
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0062945, 0.0060686
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0082775, 0.0078904
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0081543, 0.0085254
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0108256, 0.0103966
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0082196, 0.0078633
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0089913, 0.0093136
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0084571, 0.0080974
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0307863, 0.0324626

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127165, upper bound: 0.0132196
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127156, upper bound: 0.0131641
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0085236, 0.0087943
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060905, 0.0062693
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0079099, 0.0082503
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0085052, 0.0081712
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104144, 0.0108061
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078798, 0.0081968
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0092897, 0.0090142
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081090, 0.0084395
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0323727, 0.0308658

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128610, upper bound: 0.0128182
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129523, upper bound: 0.0128219
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0085042, 0.0088235
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060769, 0.0062898
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0079334, 0.0082462
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0085056, 0.0081796
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104216, 0.0108061
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078968, 0.0081953
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0092908, 0.0090205
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081260, 0.0084363
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0323750, 0.0309014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128610, upper bound: 0.0128182
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129523, upper bound: 0.0128219
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0085236, 0.0087943
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060905, 0.0062693
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0079099, 0.0082503
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0085052, 0.0081712
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104144, 0.0108061
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078798, 0.0081968
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0092897, 0.0090142
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081090, 0.0084395
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0323727, 0.0308658

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131742, upper bound: 0.0126994
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132780, upper bound: 0.0127008
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0085042, 0.0088235
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060769, 0.0062898
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0079334, 0.0082462
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0085056, 0.0081796
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104216, 0.0108061
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078968, 0.0081953
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0092908, 0.0090205
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081260, 0.0084363
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0323750, 0.0309014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131742, upper bound: 0.0126994
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132780, upper bound: 0.0127008
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0085236, 0.0087943
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060905, 0.0062693
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0079099, 0.0082503
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0085052, 0.0081712
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104144, 0.0108061
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078798, 0.0081968
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0092897, 0.0090142
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081090, 0.0084395
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0323727, 0.0308658

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128579, upper bound: 0.0128244
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129523, upper bound: 0.0128298
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0085042, 0.0088235
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060769, 0.0062898
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0079334, 0.0082462
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0085056, 0.0081796
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104216, 0.0108061
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078968, 0.0081953
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0092908, 0.0090205
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081260, 0.0084363
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0323750, 0.0309014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128579, upper bound: 0.0128244
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129523, upper bound: 0.0128298
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0085236, 0.0087943
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060905, 0.0062693
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0079099, 0.0082503
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0085052, 0.0081712
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104144, 0.0108061
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078798, 0.0081968
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0092897, 0.0090142
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081090, 0.0084395
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0323727, 0.0308658

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131593, upper bound: 0.0127014
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132780, upper bound: 0.0127039
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041294, -0.0011201, -0.0041294, -0.0011201, -0.0030093, 0.0030093
1: 0.0176409, 0.0321294, 0.0176409, 0.0321294, -0.0085042, 0.0088235
2: 0.0203480, 0.0302296, 0.0203480, 0.0302296, -0.0060769, 0.0062898
3: 0.0058547, 0.0173773, 0.0058547, 0.0173773, -0.0079334, 0.0082462
4: -0.0183173, -0.0069501, -0.0183173, -0.0069501, -0.0085056, 0.0081796
5: 0.0121191, 0.0264271, 0.0121191, 0.0264271, -0.0104216, 0.0108061
6: 0.0045073, 0.0153141, 0.0045073, 0.0153141, -0.0078968, 0.0081953
7: -0.0228774, -0.0111078, -0.0228774, -0.0111078, -0.0092908, 0.0090205
8: 0.0073552, 0.0189446, 0.0073552, 0.0189446, -0.0081260, 0.0084363
9: 0.8998641, 0.9497502, 0.8998641, 0.9497502, -0.0323750, 0.0309014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131593, upper bound: 0.0127014
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132780, upper bound: 0.0127039
time: 0.73 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.90 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0127039, upper bound: 0.0132780
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0127014, upper bound: 0.0131593
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0127039, upper bound: 0.0132780
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0127014, upper bound: 0.0131593
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0128298, upper bound: 0.0129523
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0128244, upper bound: 0.0128579
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0128298, upper bound: 0.0129523
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0128244, upper bound: 0.0128579
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0127008, upper bound: 0.0132780
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0126994, upper bound: 0.0131742
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0127008, upper bound: 0.0132780
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0126994, upper bound: 0.0131742
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0128219, upper bound: 0.0129523
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0128182, upper bound: 0.0128610
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0128219, upper bound: 0.0129523
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0128182, upper bound: 0.0128610
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0131641, upper bound: 0.0127156
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0132196, upper bound: 0.0127165
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0131641, upper bound: 0.0127156
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0132196, upper bound: 0.0127165
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0133211, upper bound: 0.0124817
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0134081, upper bound: 0.0124780
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0133211, upper bound: 0.0124817
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0134081, upper bound: 0.0124780
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0131628, upper bound: 0.0127206
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0132196, upper bound: 0.0127246
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0131628, upper bound: 0.0127206
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0132196, upper bound: 0.0127246
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0133185, upper bound: 0.0124840
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0134081, upper bound: 0.0124820
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0133185, upper bound: 0.0124840
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0134081, upper bound: 0.0124820
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0124820, upper bound: 0.0134081
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0124840, upper bound: 0.0133185
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0124820, upper bound: 0.0134081
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0124840, upper bound: 0.0133185
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0127246, upper bound: 0.0132196
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0127206, upper bound: 0.0131628
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0127246, upper bound: 0.0132196
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0127206, upper bound: 0.0131628
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0124780, upper bound: 0.0134081
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0124817, upper bound: 0.0133211
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0124780, upper bound: 0.0134081
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0124817, upper bound: 0.0133211
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0127165, upper bound: 0.0132196
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0127156, upper bound: 0.0131641
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0127165, upper bound: 0.0132196
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0127156, upper bound: 0.0131641
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0128610, upper bound: 0.0128182
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0129523, upper bound: 0.0128219
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0128610, upper bound: 0.0128182
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0129523, upper bound: 0.0128219
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0131745, upper bound: 0.0126994
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0132780, upper bound: 0.0127008
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0131745, upper bound: 0.0126994
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0132780, upper bound: 0.0127008
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0128579, upper bound: 0.0128244
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0129523, upper bound: 0.0128298
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0128579, upper bound: 0.0128244
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0129523, upper bound: 0.0128298
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0131642, upper bound: 0.0127014
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0132780, upper bound: 0.0127039
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0131642, upper bound: 0.0127014
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0132780, upper bound: 0.0127039
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0127039, upper bound: 0.0132780
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0127014, upper bound: 0.0131642
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0127039, upper bound: 0.0132780
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0127014, upper bound: 0.0131642
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0128298, upper bound: 0.0129523
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0128244, upper bound: 0.0128579
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0128298, upper bound: 0.0129523
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0128244, upper bound: 0.0128579
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0127008, upper bound: 0.0132780
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0126994, upper bound: 0.0131745
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0127008, upper bound: 0.0132780
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0126994, upper bound: 0.0131745
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0128219, upper bound: 0.0129523
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0128182, upper bound: 0.0128610
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0128219, upper bound: 0.0129523
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0128182, upper bound: 0.0128610
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0131641, upper bound: 0.0127156
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0132196, upper bound: 0.0127165
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0131641, upper bound: 0.0127156
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0132196, upper bound: 0.0127165
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0133211, upper bound: 0.0124817
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0134081, upper bound: 0.0124780
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0133211, upper bound: 0.0124817
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0134081, upper bound: 0.0124780
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0131628, upper bound: 0.0127206
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0132196, upper bound: 0.0127246
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0131628, upper bound: 0.0127206
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0132196, upper bound: 0.0127246
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0133185, upper bound: 0.0124840
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0134081, upper bound: 0.0124820
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0133185, upper bound: 0.0124840
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0134081, upper bound: 0.0124820
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0124820, upper bound: 0.0134081
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0124840, upper bound: 0.0133185
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0124820, upper bound: 0.0134081
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0124840, upper bound: 0.0133185
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0127246, upper bound: 0.0132196
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0127206, upper bound: 0.0131628
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0127246, upper bound: 0.0132196
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0127206, upper bound: 0.0131628
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0124780, upper bound: 0.0134081
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0124817, upper bound: 0.0133211
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0124780, upper bound: 0.0134081
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0124817, upper bound: 0.0133211
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0127165, upper bound: 0.0132196
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0127156, upper bound: 0.0131641
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0127165, upper bound: 0.0132196
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0127156, upper bound: 0.0131641
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0128610, upper bound: 0.0128182
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0129523, upper bound: 0.0128219
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0128610, upper bound: 0.0128182
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0129523, upper bound: 0.0128219
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0131742, upper bound: 0.0126994
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0132780, upper bound: 0.0127008
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0131742, upper bound: 0.0126994
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0132780, upper bound: 0.0127008
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0128579, upper bound: 0.0128244
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0129523, upper bound: 0.0128298
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0128579, upper bound: 0.0128244
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0129523, upper bound: 0.0128298
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0131593, upper bound: 0.0127014
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0132780, upper bound: 0.0127039
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0131593, upper bound: 0.0127014
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.90
Output dim: 9, lower bound: -0.0132780, upper bound: 0.0127039

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.76 + 350.77 = 353.52 seconds
