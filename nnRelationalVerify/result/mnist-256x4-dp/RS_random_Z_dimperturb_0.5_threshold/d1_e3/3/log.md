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
0: (-0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154)
1: (0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0097978, 0.0097978)
2: (0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0069859, 0.0069859)
3: (0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0088943, 0.0088943)
4: (-0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0092253, 0.0092253)
5: (0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0115901, 0.0115901)
6: (0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0088181, 0.0088181)
7: (-0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0099819, 0.0099819)
8: (0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0089883, 0.0089883)
9: (0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0355501, 0.0355501)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.32 + 1.58 = 2.90 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0216861, upper bound: 0.0216861

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0212727, upper bound: 0.0212560
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0212560, upper bound: 0.0212727
time: 0.67 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.39 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.39
Output dim: 9, lower bound: -0.0212727, upper bound: 0.0212560
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.39
Output dim: 9, lower bound: -0.0212560, upper bound: 0.0212727

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0096092, 0.0096214
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0068602, 0.0068687
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0087006, 0.0087204
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0090680, 0.0090486
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0113856, 0.0114079
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0086474, 0.0086656
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0098379, 0.0098248
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0088105, 0.0088270
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0348191, 0.0347344

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0211066, upper bound: 0.0207363
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0207655, upper bound: 0.0210912
time: 0.70 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0096214, 0.0096092
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0068687, 0.0068602
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0087204, 0.0087006
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0090486, 0.0090680
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0114079, 0.0113856
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0086656, 0.0086474
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0098248, 0.0098379
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0088270, 0.0088105
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0347344, 0.0348191

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0210912, upper bound: 0.0207655
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0207363, upper bound: 0.0211066
time: 0.66 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.55 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.55
Output dim: 9, lower bound: -0.0211066, upper bound: 0.0207363
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.55
Output dim: 9, lower bound: -0.0207655, upper bound: 0.0210912
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.55
Output dim: 9, lower bound: -0.0210912, upper bound: 0.0207655
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.55
Output dim: 9, lower bound: -0.0207363, upper bound: 0.0211066

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0094590, 0.0095173
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0067285, 0.0067711
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0086666, 0.0086949
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0090282, 0.0089952
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0113326, 0.0113696
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0086140, 0.0086416
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0097799, 0.0097443
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0087897, 0.0088111
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0346103, 0.0344603

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0188025, upper bound: 0.0195794
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0199350, upper bound: 0.0184902
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0095054, 0.0094713
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0067629, 0.0067371
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0086755, 0.0086865
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0090146, 0.0090114
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0113499, 0.0113548
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0086238, 0.0086322
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0097575, 0.0097682
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0087948, 0.0088062
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0345450, 0.0345383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0185067, upper bound: 0.0198920
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0196146, upper bound: 0.0187675
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0094713, 0.0095054
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0067371, 0.0067629
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0086865, 0.0086755
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0090114, 0.0090146
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0113548, 0.0113499
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0086322, 0.0086238
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0097682, 0.0097575
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0088062, 0.0087948
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0345383, 0.0345450

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0187675, upper bound: 0.0196146
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0198920, upper bound: 0.0185068
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0095173, 0.0094590
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0067711, 0.0067285
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0086949, 0.0086666
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0089952, 0.0090282
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0113696, 0.0113326
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0086416, 0.0086140
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0097443, 0.0097799
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0088111, 0.0087897
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0344603, 0.0346103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0184902, upper bound: 0.0199350
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0195794, upper bound: 0.0188025
time: 0.71 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.58 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.58
Output dim: 9, lower bound: -0.0188025, upper bound: 0.0195794
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.58
Output dim: 9, lower bound: -0.0199350, upper bound: 0.0184902
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.58
Output dim: 9, lower bound: -0.0185067, upper bound: 0.0198920
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.58
Output dim: 9, lower bound: -0.0196146, upper bound: 0.0187675
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.58
Output dim: 9, lower bound: -0.0187675, upper bound: 0.0196146
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.58
Output dim: 9, lower bound: -0.0198920, upper bound: 0.0185068
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.58
Output dim: 9, lower bound: -0.0184902, upper bound: 0.0199350
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.58
Output dim: 9, lower bound: -0.0195794, upper bound: 0.0188025

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0087274, 0.0084241
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0062024, 0.0059994
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0080893, 0.0077388
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0081254, 0.0084668
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0106890, 0.0102887
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0081061, 0.0077812
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089774, 0.0092672
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0082412, 0.0078943
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0303184, 0.0318610

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0187508, upper bound: 0.0194046
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0187519, upper bound: 0.0195218
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0083658, 0.0087838
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059569, 0.0062441
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0077104, 0.0081156
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0084955, 0.0080924
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0102518, 0.0107212
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0077536, 0.0081315
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0093005, 0.0089418
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0078728, 0.0082612
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0319930, 0.0301684

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0189228, upper bound: 0.0164531
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0176991, upper bound: 0.0174654
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0087683, 0.0083781
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0062330, 0.0059654
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0080971, 0.0077303
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0081118, 0.0084795
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0107031, 0.0102740
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0081147, 0.0077718
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089550, 0.0092865
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0082457, 0.0078893
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0302531, 0.0319228

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0169059, upper bound: 0.0186253
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0172135, upper bound: 0.0183196
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0084122, 0.0087441
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059912, 0.0062144
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0077193, 0.0081073
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0084830, 0.0081086
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0102690, 0.0107076
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0077634, 0.0081230
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0092813, 0.0089657
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0078779, 0.0082566
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0319336, 0.0302464

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0185740, upper bound: 0.0166548
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0174519, upper bound: 0.0177305
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0087441, 0.0084122
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0062144, 0.0059912
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0081073, 0.0077193
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0081086, 0.0084830
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0107076, 0.0102690
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0081230, 0.0077634
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089657, 0.0092813
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0082566, 0.0078779
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0302464, 0.0319336

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0164727, upper bound: 0.0171982
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163992, upper bound: 0.0172308
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0083781, 0.0087683
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059654, 0.0062330
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0077303, 0.0080971
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0084795, 0.0081118
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0102740, 0.0107031
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0077718, 0.0081147
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0092865, 0.0089550
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0078893, 0.0082457
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0319228, 0.0302531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0183196, upper bound: 0.0172135
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0186253, upper bound: 0.0169059
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0087838, 0.0083659
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0062441, 0.0059569
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0081156, 0.0077104
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0080924, 0.0084955
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0107212, 0.0102518
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0081315, 0.0077536
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089418, 0.0093005
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0082612, 0.0078728
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0301684, 0.0319930

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0157967, upper bound: 0.0156959
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0154975, upper bound: 0.0160405
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0084241, 0.0087274
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059994, 0.0062024
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0077388, 0.0080893
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0084668, 0.0081254
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0102887, 0.0106890
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0077812, 0.0081061
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0092672, 0.0089774
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0078943, 0.0082412
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0318611, 0.0303184

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 131

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0195218, upper bound: 0.0187519
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0194046, upper bound: 0.0187508
time: 0.69 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.62 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.62
Output dim: 9, lower bound: -0.0187508, upper bound: 0.0194046
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.62
Output dim: 9, lower bound: -0.0187519, upper bound: 0.0195218
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.62
Output dim: 9, lower bound: -0.0189228, upper bound: 0.0164531
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.62
Output dim: 9, lower bound: -0.0176991, upper bound: 0.0174654
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.62
Output dim: 9, lower bound: -0.0169059, upper bound: 0.0186253
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.62
Output dim: 9, lower bound: -0.0172135, upper bound: 0.0183196
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.62
Output dim: 9, lower bound: -0.0185740, upper bound: 0.0166548
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.62
Output dim: 9, lower bound: -0.0174519, upper bound: 0.0177305
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.62
Output dim: 9, lower bound: -0.0164727, upper bound: 0.0171982
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.62
Output dim: 9, lower bound: -0.0163992, upper bound: 0.0172308
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.62
Output dim: 9, lower bound: -0.0183196, upper bound: 0.0172135
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.62
Output dim: 9, lower bound: -0.0186253, upper bound: 0.0169059
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.62
Output dim: 9, lower bound: -0.0157967, upper bound: 0.0156959
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.62
Output dim: 9, lower bound: -0.0154975, upper bound: 0.0160405
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.62
Output dim: 9, lower bound: -0.0195218, upper bound: 0.0187519
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.62
Output dim: 9, lower bound: -0.0194046, upper bound: 0.0187508

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0085813, 0.0083060
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0061058, 0.0059211
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0079044, 0.0076052
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0080610, 0.0083580
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0105567, 0.0102094
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0079816, 0.0077024
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089852, 0.0092407
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0080054, 0.0077198
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0296830, 0.0310283

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0158447, upper bound: 0.0155664
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0154996, upper bound: 0.0158586
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0086093, 0.0083033
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0061241, 0.0059205
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0079556, 0.0076177
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0080657, 0.0084023
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0106097, 0.0102112
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0080273, 0.0077080
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089899, 0.0092750
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0080667, 0.0077323
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0297105, 0.0312256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0173627, upper bound: 0.0179078
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0173627, upper bound: 0.0179078
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0077607, 0.0083328
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0055651, 0.0059523
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0072410, 0.0077675
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0081677, 0.0076250
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0096499, 0.0102885
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0073160, 0.0078152
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0090885, 0.0086190
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0073013, 0.0078161
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0302340, 0.0277830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163174, upper bound: 0.0138834
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163174, upper bound: 0.0145198
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0079110, 0.0081786
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0056633, 0.0058524
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0073637, 0.0076461
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0080281, 0.0077668
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0098199, 0.0101193
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0074386, 0.0076939
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089776, 0.0087321
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0074288, 0.0076897
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0296076, 0.0284195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0161978, upper bound: 0.0161438
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0164432, upper bound: 0.0158257
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0086851, 0.0082427
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0061254, 0.0058231
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0072885, 0.0067718
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0071584, 0.0076433
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0098463, 0.0092811
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0072604, 0.0067901
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0081424, 0.0085652
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0077113, 0.0072196
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0269678, 0.0291605

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0142323, upper bound: 0.0144481
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138848, upper bound: 0.0148068
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0086374, 0.0082948
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0060936, 0.0058579
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0071700, 0.0069217
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0072756, 0.0075467
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0097381, 0.0094172
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0071600, 0.0069175
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0082336, 0.0084804
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0076068, 0.0073548
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0274908, 0.0287177

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0158502, upper bound: 0.0167856
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0158502, upper bound: 0.0167856
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0078070, 0.0082911
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0055995, 0.0059214
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0072498, 0.0077624
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0081570, 0.0076412
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0096672, 0.0102781
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0073258, 0.0078094
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0090694, 0.0086428
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0073064, 0.0078148
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0301813, 0.0278610

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0169908, upper bound: 0.0153133
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0169908, upper bound: 0.0153133
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0079511, 0.0081389
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0056924, 0.0058227
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0073690, 0.0076379
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0080156, 0.0077767
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0098297, 0.0101057
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0074445, 0.0076854
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089584, 0.0087504
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0074304, 0.0076851
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0295482, 0.0284678

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0154763, upper bound: 0.0152353
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0149964, upper bound: 0.0152579
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0085013, 0.0081037
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0060714, 0.0058035
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0077433, 0.0073577
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0077702, 0.0081630
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0103100, 0.0098590
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0077999, 0.0074399
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0086629, 0.0090243
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0078855, 0.0075334
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0287421, 0.0305321

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0141975, upper bound: 0.0147350
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0141975, upper bound: 0.0147350
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0087441, 0.0081694
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0062144, 0.0058481
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0081073, 0.0073553
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0077886, 0.0084830
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0107076, 0.0098715
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0081230, 0.0074404
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0087087, 0.0092813
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0082566, 0.0075068
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0288448, 0.0319336

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0155120, upper bound: 0.0163699
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0155163, upper bound: 0.0162565
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0082948, 0.0086374
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058579, 0.0060936
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0069217, 0.0071700
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0075467, 0.0072756
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0094172, 0.0097381
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0069175, 0.0071600
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0084804, 0.0082336
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0073548, 0.0076068
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0287177, 0.0274908

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0172851, upper bound: 0.0152117
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0161546, upper bound: 0.0161886
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0082427, 0.0086851
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058231, 0.0061254
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0067718, 0.0072885
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0076433, 0.0071584
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0092811, 0.0098463
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0067901, 0.0072604
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0085652, 0.0081424
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0072196, 0.0077113
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0291606, 0.0269678

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0148068, upper bound: 0.0138848
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0144481, upper bound: 0.0142323
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0087094, 0.0084598
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0061812, 0.0060109
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0080229, 0.0077597
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0081452, 0.0083934
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0106118, 0.0103190
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0080382, 0.0077954
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0090065, 0.0092136
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0081776, 0.0079184
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0304700, 0.0315832

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0142323, upper bound: 0.0144558
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0145651, upper bound: 0.0140887
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0087838, 0.0082915
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0062441, 0.0058940
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0081156, 0.0076177
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0079903, 0.0084955
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0107212, 0.0101424
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0081315, 0.0076603
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0088550, 0.0093005
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0082612, 0.0077892
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0297585, 0.0319930

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138723, upper bound: 0.0148068
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0142604, upper bound: 0.0145244
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0083033, 0.0086093
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059205, 0.0061241
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0076177, 0.0079556
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0084023, 0.0080657
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0102112, 0.0106097
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0077080, 0.0080273
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0092750, 0.0089899
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0077323, 0.0080667
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0312256, 0.0297105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171723, upper bound: 0.0163576
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171250, upper bound: 0.0164223
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0083060, 0.0085813
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059211, 0.0061058
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0076052, 0.0079044
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0083580, 0.0080610
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0102094, 0.0105567
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0077024, 0.0079816
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0092407, 0.0089852
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0077198, 0.0080054
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0310283, 0.0296830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0175100, upper bound: 0.0174913
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0181058, upper bound: 0.0172795
time: 0.73 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.76 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 9, lower bound: -0.0158447, upper bound: 0.0155664
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 9, lower bound: -0.0154996, upper bound: 0.0158586
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 9, lower bound: -0.0173627, upper bound: 0.0179078
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 9, lower bound: -0.0173627, upper bound: 0.0179078
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 9, lower bound: -0.0163174, upper bound: 0.0138834
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 9, lower bound: -0.0163174, upper bound: 0.0145198
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 9, lower bound: -0.0161978, upper bound: 0.0161438
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 9, lower bound: -0.0164432, upper bound: 0.0158257
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 9, lower bound: -0.0142323, upper bound: 0.0144481
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 9, lower bound: -0.0138848, upper bound: 0.0148068
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 9, lower bound: -0.0158502, upper bound: 0.0167856
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 9, lower bound: -0.0158502, upper bound: 0.0167856
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 9, lower bound: -0.0169908, upper bound: 0.0153133
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 9, lower bound: -0.0169908, upper bound: 0.0153133
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 9, lower bound: -0.0154763, upper bound: 0.0152353
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 9, lower bound: -0.0149964, upper bound: 0.0152579
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 9, lower bound: -0.0141975, upper bound: 0.0147350
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 9, lower bound: -0.0141975, upper bound: 0.0147350
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 9, lower bound: -0.0155120, upper bound: 0.0163699
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 9, lower bound: -0.0155163, upper bound: 0.0162565
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 9, lower bound: -0.0172851, upper bound: 0.0152117
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 9, lower bound: -0.0161546, upper bound: 0.0161886
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 9, lower bound: -0.0148068, upper bound: 0.0138848
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 9, lower bound: -0.0144481, upper bound: 0.0142323
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 9, lower bound: -0.0142323, upper bound: 0.0144558
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 9, lower bound: -0.0145651, upper bound: 0.0140887
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 9, lower bound: -0.0138723, upper bound: 0.0148068
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 9, lower bound: -0.0142604, upper bound: 0.0145244
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 9, lower bound: -0.0171723, upper bound: 0.0163576
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 9, lower bound: -0.0171250, upper bound: 0.0164223
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 9, lower bound: -0.0175100, upper bound: 0.0174913
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 9, lower bound: -0.0181058, upper bound: 0.0172795

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0086531, 0.0085254
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0061395, 0.0060580
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0079966, 0.0077900
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0081776, 0.0083647
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0105796, 0.0103550
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0080128, 0.0078235
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0090456, 0.0091804
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0081576, 0.0079451
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0306190, 0.0314512

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0143757, upper bound: 0.0143285
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0146194, upper bound: 0.0138943
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0087274, 0.0083498
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0062024, 0.0059365
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0080893, 0.0076461
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0080233, 0.0084668
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0106890, 0.0101794
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0081061, 0.0076879
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0088905, 0.0092672
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0082412, 0.0078107
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0299086, 0.0318610

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 131

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0144326, upper bound: 0.0141491
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139024, upper bound: 0.0148072
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0087096, 0.0084020
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0061895, 0.0059837
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0080686, 0.0077080
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0081145, 0.0084599
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0106831, 0.0102793
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0080924, 0.0077594
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089707, 0.0092622
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0082260, 0.0078700
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0302697, 0.0318310

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0158648, upper bound: 0.0165645
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0159966, upper bound: 0.0162401
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0087053, 0.0084241
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0061866, 0.0059994
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0080893, 0.0077181
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0081186, 0.0084668
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0106890, 0.0102828
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0081061, 0.0077675
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089723, 0.0092672
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0082412, 0.0078791
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0302884, 0.0318610

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163572, upper bound: 0.0158701
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0151587, upper bound: 0.0168779
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0081231, 0.0084527
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058138, 0.0060419
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0073464, 0.0077293
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0081283, 0.0077725
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0098542, 0.0102789
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0074305, 0.0077837
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089669, 0.0086848
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0075016, 0.0078887
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0303557, 0.0287668

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162581, upper bound: 0.0138316
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160115, upper bound: 0.0138304
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0083658, 0.0085410
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059569, 0.0061011
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0077104, 0.0077516
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0081755, 0.0080924
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0102518, 0.0103236
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0077536, 0.0078085
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0090435, 0.0089418
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0078728, 0.0078900
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0305915, 0.0301684

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126038, upper bound: 0.0114365
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126038, upper bound: 0.0114365
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0082826, 0.0086522
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058493, 0.0061043
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0069018, 0.0071790
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0075551, 0.0072562
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0093950, 0.0097473
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0068993, 0.0071682
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0084922, 0.0082205
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0073383, 0.0076145
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0287568, 0.0274061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0142870, upper bound: 0.0137270
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138240, upper bound: 0.0137555
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0082294, 0.0087005
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058137, 0.0061366
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0067579, 0.0073070
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0076593, 0.0071438
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0092650, 0.0098644
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0067763, 0.0072772
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0085792, 0.0081332
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0072042, 0.0077267
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0292308, 0.0269018

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163840, upper bound: 0.0157730
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0159747, upper bound: 0.0157310
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0086939, 0.0084683
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0061701, 0.0060162
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0080044, 0.0077716
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0081518, 0.0083774
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0105937, 0.0103271
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0080213, 0.0078050
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0090121, 0.0091996
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0081622, 0.0079314
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0304957, 0.0315129

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131821, upper bound: 0.0134071
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131797, upper bound: 0.0134051
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0087683, 0.0083037
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0062330, 0.0059025
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0080971, 0.0076376
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0080097, 0.0084795
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0107031, 0.0101646
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0081147, 0.0076785
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0088681, 0.0092865
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0082457, 0.0078057
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0298432, 0.0319228

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127881, upper bound: 0.0130909
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0123762, upper bound: 0.0137512
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0087547, 0.0083560
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0062234, 0.0059497
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0080764, 0.0077025
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0081018, 0.0084727
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0106971, 0.0102655
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0081009, 0.0077527
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089488, 0.0092814
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0082306, 0.0078671
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0302093, 0.0318928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0148042, upper bound: 0.0146822
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138740, upper bound: 0.0157210
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0087462, 0.0083781
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0062172, 0.0059654
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0080971, 0.0077096
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0081049, 0.0084795
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0107031, 0.0102681
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0081147, 0.0077580
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089499, 0.0092865
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0082457, 0.0078741
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0302230, 0.0319228

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127477, upper bound: 0.0135051
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127477, upper bound: 0.0135051
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0084043, 0.0087220
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059859, 0.0061987
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0076986, 0.0080894
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0084758, 0.0081017
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0102631, 0.0107014
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0077497, 0.0081101
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0092761, 0.0089606
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0078628, 0.0082413
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0319022, 0.0302164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0169458, upper bound: 0.0151231
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0168789, upper bound: 0.0152717
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0083901, 0.0087441
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059755, 0.0062144
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0077193, 0.0080867
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0084761, 0.0081086
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0102690, 0.0107016
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0077634, 0.0081092
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0092763, 0.0089657
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0078779, 0.0082415
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0319036, 0.0302464

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0152047, upper bound: 0.0139978
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0156023, upper bound: 0.0139503
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0081694, 0.0084129
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058481, 0.0060128
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0073553, 0.0077205
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0081161, 0.0077886
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0098715, 0.0102647
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0074404, 0.0077743
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089477, 0.0087087
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0075068, 0.0078830
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0303015, 0.0288448

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139030, upper bound: 0.0140130
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0142433, upper bound: 0.0139130
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0084122, 0.0085013
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059912, 0.0060714
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0077193, 0.0077433
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0081630, 0.0081086
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0102690, 0.0103100
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0077634, 0.0077999
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0090243, 0.0089657
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0078779, 0.0078855
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0305321, 0.0302464

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0120452, upper bound: 0.0119719
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0120452, upper bound: 0.0119719
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0087261, 0.0083901
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0062013, 0.0059755
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0080867, 0.0076877
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0080981, 0.0084761
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0107016, 0.0102598
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0081092, 0.0077421
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089581, 0.0092763
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0082415, 0.0078523
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0301990, 0.0319036

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
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128132, upper bound: 0.0133700
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128626, upper bound: 0.0131312
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0087220, 0.0084122
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0061987, 0.0059912
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0081073, 0.0076986
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0081017, 0.0084830
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0107076, 0.0102631
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0081230, 0.0077497
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089606, 0.0092813
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0082566, 0.0078628
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0302164, 0.0319336

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 131

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0141642, upper bound: 0.0145797
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0141207, upper bound: 0.0146977
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0083730, 0.0079781
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0060093, 0.0057454
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0076069, 0.0071501
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0076101, 0.0080527
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0101688, 0.0096495
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0076871, 0.0072632
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0086002, 0.0089754
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0077155, 0.0072739
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0277904, 0.0297858

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 131

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0154624, upper bound: 0.0160955
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0154630, upper bound: 0.0163128
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0082373, 0.0080411
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059212, 0.0057861
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0074582, 0.0072189
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0076783, 0.0078992
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0099885, 0.0097303
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0075447, 0.0073275
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0086598, 0.0088512
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0075691, 0.0073368
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0280986, 0.0290935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134440, upper bound: 0.0137853
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134440, upper bound: 0.0137853
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0077729, 0.0083101
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0055737, 0.0059368
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0072609, 0.0077450
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0081480, 0.0076444
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0096721, 0.0102652
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0073342, 0.0077967
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0090695, 0.0086321
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0073178, 0.0077962
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0301448, 0.0278677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0134430, upper bound: 0.0126591
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129734, upper bound: 0.0128495
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0079275, 0.0081631
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0056733, 0.0058412
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0073943, 0.0076276
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0080121, 0.0077896
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0098479, 0.0101012
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0074671, 0.0076771
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089636, 0.0087438
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0074599, 0.0076742
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0295374, 0.0285195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 131

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160932, upper bound: 0.0161362
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0156170, upper bound: 0.0160412
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0083037, 0.0089104
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059025, 0.0063199
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0076376, 0.0082304
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0085976, 0.0080097
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0101646, 0.0108480
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0076785, 0.0082302
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0093982, 0.0088681
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0078057, 0.0083691
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0325116, 0.0298432

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137512, upper bound: 0.0123762
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0130909, upper bound: 0.0127881
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0083781, 0.0086939
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059654, 0.0061701
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0077303, 0.0080044
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0083774, 0.0081118
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0102740, 0.0105937
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0077718, 0.0080213
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0091996, 0.0089550
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0078893, 0.0081622
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0315129, 0.0302531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 131

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0143942, upper bound: 0.0141882
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0144027, upper bound: 0.0141777
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0087005, 0.0082294
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0061366, 0.0058137
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0073070, 0.0067579
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0071438, 0.0076593
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0098644, 0.0092650
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0072772, 0.0067763
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0081332, 0.0085792
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0077267, 0.0072042
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0269018, 0.0292308

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131794, upper bound: 0.0128734
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126352, upper bound: 0.0134016
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0086522, 0.0082826
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0061043, 0.0058493
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0071790, 0.0069018
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0072562, 0.0075551
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0097473, 0.0093950
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0071682, 0.0068993
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0082205, 0.0084922
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0076145, 0.0073383
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0274061, 0.0287568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 131

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0144323, upper bound: 0.0140429
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0145222, upper bound: 0.0140433
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0087005, 0.0082294
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0061366, 0.0058137
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0073070, 0.0067579
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0071438, 0.0076593
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0098644, 0.0092650
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0072772, 0.0067763
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0081332, 0.0085792
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0077267, 0.0072042
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0269018, 0.0292308

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127764, upper bound: 0.0130909
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0123737, upper bound: 0.0137512
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0086522, 0.0082826
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0061043, 0.0058493
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0071790, 0.0069018
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0072562, 0.0075551
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0097473, 0.0093950
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0071682, 0.0068993
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0082205, 0.0084922
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0076145, 0.0073383
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0274061, 0.0287568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 131

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0141248, upper bound: 0.0144715
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0142174, upper bound: 0.0144771
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0081814, 0.0084081
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058564, 0.0060083
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0073748, 0.0077190
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0081129, 0.0078055
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0098912, 0.0102623
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0074582, 0.0077728
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089443, 0.0087204
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0075231, 0.0078824
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0302860, 0.0289169

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 131

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0159626, upper bound: 0.0141218
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0154118, upper bound: 0.0151894
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0084241, 0.0084847
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059994, 0.0060593
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0077388, 0.0077253
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0081468, 0.0081254
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0102887, 0.0102915
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0077812, 0.0077831
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0090102, 0.0089774
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0078943, 0.0078700
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0304595, 0.0303184

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0154828, upper bound: 0.0152537
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0159373, upper bound: 0.0151776
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0083409, 0.0085848
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058919, 0.0060553
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0069302, 0.0071372
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0075146, 0.0072892
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0094320, 0.0097037
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0069270, 0.0071316
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0084435, 0.0082561
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0073598, 0.0075813
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0285676, 0.0275562

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0161262, upper bound: 0.0160389
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0161262, upper bound: 0.0160389
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0083034, 0.0086442
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058669, 0.0060948
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0068168, 0.0072807
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0076306, 0.0072016
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0093291, 0.0098323
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0068305, 0.0072519
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0085459, 0.0081923
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0072565, 0.0077067
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0290988, 0.0271677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0157640, upper bound: 0.0150874
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0157587, upper bound: 0.0151994
time: 0.75 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.69 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0143757, upper bound: 0.0143285
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0146194, upper bound: 0.0138943
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0144326, upper bound: 0.0141491
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0139024, upper bound: 0.0148072
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0158648, upper bound: 0.0165645
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0159966, upper bound: 0.0162401
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0163572, upper bound: 0.0158701
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0151587, upper bound: 0.0168779
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0162581, upper bound: 0.0138316
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0160115, upper bound: 0.0138304
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0126038, upper bound: 0.0114365
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0126038, upper bound: 0.0114365
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0142870, upper bound: 0.0137270
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0138240, upper bound: 0.0137555
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0163840, upper bound: 0.0157730
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0159747, upper bound: 0.0157310
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0131821, upper bound: 0.0134071
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0131797, upper bound: 0.0134051
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0127881, upper bound: 0.0130909
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0123762, upper bound: 0.0137512
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0148042, upper bound: 0.0146822
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0138740, upper bound: 0.0157210
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0127477, upper bound: 0.0135051
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0127477, upper bound: 0.0135051
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0169458, upper bound: 0.0151231
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0168789, upper bound: 0.0152717
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0152047, upper bound: 0.0139978
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0156023, upper bound: 0.0139503
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0139030, upper bound: 0.0140130
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0142433, upper bound: 0.0139130
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0120452, upper bound: 0.0119719
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0120452, upper bound: 0.0119719
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0128132, upper bound: 0.0133700
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0128626, upper bound: 0.0131312
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0141642, upper bound: 0.0145797
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0141207, upper bound: 0.0146977
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0154624, upper bound: 0.0160955
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0154630, upper bound: 0.0163128
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0134440, upper bound: 0.0137853
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0134440, upper bound: 0.0137853
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0134430, upper bound: 0.0126591
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0129734, upper bound: 0.0128495
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0160932, upper bound: 0.0161362
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0156170, upper bound: 0.0160412
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0137512, upper bound: 0.0123762
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0130909, upper bound: 0.0127881
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0143942, upper bound: 0.0141882
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0144027, upper bound: 0.0141777
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0131794, upper bound: 0.0128734
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0126352, upper bound: 0.0134016
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0144323, upper bound: 0.0140429
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0145222, upper bound: 0.0140433
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0127764, upper bound: 0.0130909
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0123737, upper bound: 0.0137512
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0141248, upper bound: 0.0144715
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0142174, upper bound: 0.0144771
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0159626, upper bound: 0.0141218
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0154118, upper bound: 0.0151894
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0154828, upper bound: 0.0152537
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0159373, upper bound: 0.0151776
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0161262, upper bound: 0.0160389
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0161262, upper bound: 0.0160389
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0157640, upper bound: 0.0150874
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 9, lower bound: -0.0157587, upper bound: 0.0151994

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0086442, 0.0083034
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0060948, 0.0058669
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0072807, 0.0068168
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0072016, 0.0076306
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0098323, 0.0093291
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0072519, 0.0068305
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0081923, 0.0085459
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0077067, 0.0072565
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0271677, 0.0290988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 131

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133271, upper bound: 0.0132904
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133268, upper bound: 0.0132867
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0085848, 0.0083409
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0060553, 0.0058919
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0071372, 0.0069302
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0072892, 0.0075146
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0097037, 0.0094320
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0071316, 0.0069270
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0082561, 0.0084435
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0075813, 0.0073598
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0275562, 0.0285676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 131

### Candidate
type: RSZ, layer: 3, pos: 133

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0135580, upper bound: 0.0128587
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0135580, upper bound: 0.0128587
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0081223, 0.0079663
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058106, 0.0057026
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0076198, 0.0073989
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0077983, 0.0079994
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0100872, 0.0098568
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0076685, 0.0074721
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0087626, 0.0089444
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0076697, 0.0074612
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0285611, 0.0294756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0134260, upper bound: 0.0131022
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0134023, upper bound: 0.0130906
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0082682, 0.0078190
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059054, 0.0056077
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0077401, 0.0072693
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0076580, 0.0081376
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0102550, 0.0096869
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0077910, 0.0073436
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0086545, 0.0090504
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0077948, 0.0073228
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0279330, 0.0300928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0124476, upper bound: 0.0135623
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126857, upper bound: 0.0131880
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0086442, 0.0083034
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0060948, 0.0058669
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0072807, 0.0068168
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0072016, 0.0076306
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0098323, 0.0093291
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0072519, 0.0068305
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0081923, 0.0085459
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0077067, 0.0072565
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0271677, 0.0290988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Candidate
type: RSZ, layer: 3, pos: 131

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127160, upper bound: 0.0133331
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127160, upper bound: 0.0133331
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0085848, 0.0083409
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0060553, 0.0058919
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0071372, 0.0069302
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0072892, 0.0075146
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0097037, 0.0094320
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0071316, 0.0069270
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0082561, 0.0084435
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0075813, 0.0073598
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0275562, 0.0285676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 131

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0151858, upper bound: 0.0154000
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0151910, upper bound: 0.0151810
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0081223, 0.0079663
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058106, 0.0057026
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0076198, 0.0073989
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0077983, 0.0079994
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0100872, 0.0098568
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0076685, 0.0074721
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0087626, 0.0089444
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0076697, 0.0074612
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0285611, 0.0294756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0148301, upper bound: 0.0145404
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0149751, upper bound: 0.0143124
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0082682, 0.0078190
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059054, 0.0056077
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0077401, 0.0072693
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0076580, 0.0081376
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0102550, 0.0096869
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0077910, 0.0073436
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0086545, 0.0090504
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0077948, 0.0073228
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0279330, 0.0300928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0143368, upper bound: 0.0160553
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0143396, upper bound: 0.0158514
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0082560, 0.0086656
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058850, 0.0061658
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0075960, 0.0079819
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0084310, 0.0080392
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0101826, 0.0106418
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0076895, 0.0080527
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0093082, 0.0089614
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0077076, 0.0080867
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0313576, 0.0295853

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0125642, upper bound: 0.0112900
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0125642, upper bound: 0.0112900
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0082477, 0.0086337
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058785, 0.0061446
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0075768, 0.0079260
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0083791, 0.0080279
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0101724, 0.0105797
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0076748, 0.0079988
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0092709, 0.0089496
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0076983, 0.0080295
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0311270, 0.0295329

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0150528, upper bound: 0.0129617
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0151887, upper bound: 0.0129169
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0081231, 0.0084527
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058138, 0.0060419
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0073464, 0.0077293
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0081283, 0.0077725
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0098542, 0.0102789
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0074305, 0.0077837
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089669, 0.0086848
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0075016, 0.0078887
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0303557, 0.0287668

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0108162, upper bound: 0.0105294
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0108162, upper bound: 0.0105294
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0083658, 0.0085410
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059569, 0.0061011
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0077104, 0.0077516
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0081755, 0.0080924
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0102518, 0.0103236
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0077536, 0.0078085
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0090435, 0.0089418
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0078728, 0.0078900
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0305915, 0.0301684

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0108033, upper bound: 0.0105294
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0108033, upper bound: 0.0105294
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0082560, 0.0086656
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058850, 0.0061658
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0075960, 0.0079819
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0084310, 0.0080392
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0101826, 0.0106418
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0076895, 0.0080527
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0093082, 0.0089614
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0077076, 0.0080867
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0313576, 0.0295853

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147851, upper bound: 0.0145243
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147851, upper bound: 0.0145243
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0082477, 0.0086337
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058785, 0.0061446
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0075768, 0.0079260
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0083791, 0.0080279
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0101724, 0.0105797
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0076748, 0.0079988
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0092709, 0.0089496
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0076983, 0.0080295
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0311270, 0.0295329

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0149598, upper bound: 0.0148402
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0150988, upper bound: 0.0148225
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0083101, 0.0077729
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059368, 0.0055737
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0077450, 0.0072609
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0076444, 0.0081480
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0102652, 0.0096721
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0077967, 0.0073342
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0086321, 0.0090695
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0077962, 0.0073178
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0278677, 0.0301448

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0123295, upper bound: 0.0137055
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0123315, upper bound: 0.0136923
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0081631, 0.0079275
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058412, 0.0056733
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0076276, 0.0073943
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0077896, 0.0080121
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0101012, 0.0098479
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0076771, 0.0074671
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0087438, 0.0089636
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0076742, 0.0074599
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0285195, 0.0295374

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
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0105294, upper bound: 0.0107950
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0105294, upper bound: 0.0108162
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0083101, 0.0077729
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059368, 0.0055737
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0077450, 0.0072609
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0076444, 0.0081480
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0102652, 0.0096721
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0077967, 0.0073342
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0086321, 0.0090695
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0077962, 0.0073178
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0278677, 0.0301448

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138358, upper bound: 0.0152980
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137757, upper bound: 0.0156758
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0082921, 0.0086259
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059128, 0.0061361
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0075978, 0.0079737
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0084185, 0.0080433
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0101861, 0.0106282
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0076900, 0.0080442
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0092891, 0.0089752
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0077118, 0.0080822
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0312982, 0.0296075

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0151619, upper bound: 0.0137790
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0155588, upper bound: 0.0136737
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0082940, 0.0086016
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059129, 0.0061201
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0075857, 0.0079243
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0083748, 0.0080441
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0101897, 0.0105765
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0076846, 0.0079984
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0092582, 0.0089735
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0077035, 0.0080252
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0311039, 0.0296109

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0158851, upper bound: 0.0144547
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160435, upper bound: 0.0144517
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0083289, 0.0085994
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058836, 0.0060658
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0069107, 0.0071473
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0075242, 0.0072724
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0094122, 0.0097142
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0069092, 0.0071403
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0084563, 0.0082444
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0073435, 0.0075896
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0286129, 0.0274841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0108474, upper bound: 0.0101706
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0108474, upper bound: 0.0102176
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0082917, 0.0086608
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058587, 0.0061069
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0068031, 0.0072987
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0076468, 0.0071904
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0093161, 0.0098508
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0068179, 0.0072687
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0085600, 0.0081809
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0072402, 0.0077222
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0291714, 0.0271156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 180

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0146045, upper bound: 0.0131310
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147750, upper bound: 0.0131261
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0083289, 0.0085994
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058836, 0.0060658
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0069107, 0.0071473
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0075242, 0.0072724
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0094122, 0.0097142
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0069092, 0.0071403
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0084563, 0.0082444
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0073435, 0.0075896
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0286129, 0.0274841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138476, upper bound: 0.0139623
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135600, upper bound: 0.0139620
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0082917, 0.0086608
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058587, 0.0061069
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0068031, 0.0072987
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0076468, 0.0071904
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0093161, 0.0098508
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0068179, 0.0072687
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0085600, 0.0081809
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0072402, 0.0077222
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0291714, 0.0271156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0106502, upper bound: 0.0105709
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0106502, upper bound: 0.0105709
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0086016, 0.0082940
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0061201, 0.0059129
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0079243, 0.0075857
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0080441, 0.0083748
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0105765, 0.0101897
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0079984, 0.0076846
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089735, 0.0092582
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0080252, 0.0077035
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0296109, 0.0311039

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0119352, upper bound: 0.0117163
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0115276, upper bound: 0.0122236
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0086259, 0.0082921
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0061361, 0.0059128
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0079737, 0.0075978
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0080433, 0.0084185
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0106282, 0.0101861
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0080442, 0.0076900
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089752, 0.0092891
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0080822, 0.0077118
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0296075, 0.0312982

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133391, upper bound: 0.0139349
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133624, upper bound: 0.0137488
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0086016, 0.0082940
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0061201, 0.0059129
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0079243, 0.0075857
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0080441, 0.0083748
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0105765, 0.0101897
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0079984, 0.0076846
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089735, 0.0092582
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0080252, 0.0077035
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0296109, 0.0311039

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 43

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133790, upper bound: 0.0138234
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133790, upper bound: 0.0138234
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0086259, 0.0082921
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0061361, 0.0059128
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0079737, 0.0075978
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0080433, 0.0084185
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0106282, 0.0101861
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0080442, 0.0076900
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089752, 0.0092891
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0080822, 0.0077118
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0296075, 0.0312982

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0143555, upper bound: 0.0146378
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130916, upper bound: 0.0151520
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0087261, 0.0083901
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0062013, 0.0059755
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0080867, 0.0076877
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0080981, 0.0084761
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0107016, 0.0102598
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0081092, 0.0077421
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089581, 0.0092763
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0082415, 0.0078523
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0301990, 0.0319036

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0120445, upper bound: 0.0124058
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0121000, upper bound: 0.0121283
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0087220, 0.0084122
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0061987, 0.0059912
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0081073, 0.0076986
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0081017, 0.0084830
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0107076, 0.0102631
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0081230, 0.0077497
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089606, 0.0092813
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0082566, 0.0078628
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0302164, 0.0319336

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0112607, upper bound: 0.0111165
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0107944, upper bound: 0.0114688
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0082691, 0.0086501
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058954, 0.0061546
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0076163, 0.0079635
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0084151, 0.0080616
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0102079, 0.0106237
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0077076, 0.0080358
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0092942, 0.0089780
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0077285, 0.0080713
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0312873, 0.0296884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0146370, upper bound: 0.0147654
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0146370, upper bound: 0.0147654
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0082599, 0.0086157
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058871, 0.0061321
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0075967, 0.0079063
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0083614, 0.0080473
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0101946, 0.0105597
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0076930, 0.0079821
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0092540, 0.0089627
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0077148, 0.0080098
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0310465, 0.0296176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137550, upper bound: 0.0136482
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132494, upper bound: 0.0136796
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0077729, 0.0083101
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0055737, 0.0059368
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0072609, 0.0077450
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0081480, 0.0076444
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0096721, 0.0102652
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0073342, 0.0077967
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0090695, 0.0086321
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0073178, 0.0077962
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0301448, 0.0278677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126666, upper bound: 0.0113386
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126786, upper bound: 0.0113523
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0082691, 0.0086501
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058954, 0.0061546
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0076163, 0.0079635
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0084151, 0.0080616
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0102079, 0.0106237
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0077076, 0.0080358
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0092942, 0.0089780
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0077285, 0.0080713
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0312873, 0.0296884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133557, upper bound: 0.0131356
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133571, upper bound: 0.0131381
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0082599, 0.0086157
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058871, 0.0061321
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0075967, 0.0079063
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0083614, 0.0080473
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0101946, 0.0105597
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0076930, 0.0079821
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0092540, 0.0089627
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0077148, 0.0080098
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0310465, 0.0296176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 133

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133439, upper bound: 0.0125896
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128008, upper bound: 0.0130978
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0086337, 0.0082477
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0061446, 0.0058785
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0079260, 0.0075768
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0080279, 0.0083791
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0105797, 0.0101724
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0079988, 0.0076748
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089496, 0.0092709
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0080295, 0.0076983
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0295329, 0.0311270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133712, upper bound: 0.0125187
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128040, upper bound: 0.0129572
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0086656, 0.0082560
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0061658, 0.0058850
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0079819, 0.0075960
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0080392, 0.0084310
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0106418, 0.0101826
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0080527, 0.0076895
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089614, 0.0093082
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0080867, 0.0077076
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0295853, 0.0313576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0134815, upper bound: 0.0125501
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128046, upper bound: 0.0129514
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0083328, 0.0077607
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059523, 0.0055651
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0077675, 0.0072410
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0076250, 0.0081677
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0102885, 0.0096499
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0078152, 0.0073160
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0086190, 0.0090885
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0078161, 0.0073013
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0277830, 0.0302340

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0113487, upper bound: 0.0126786
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0113353, upper bound: 0.0126666
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0086337, 0.0082477
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0061446, 0.0058785
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0079260, 0.0075768
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0080279, 0.0083791
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0105797, 0.0101724
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0079988, 0.0076748
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089496, 0.0092709
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0080295, 0.0076983
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0295329, 0.0311270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0130991, upper bound: 0.0134167
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0130893, upper bound: 0.0134158
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0086656, 0.0082560
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0061658, 0.0058850
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0079819, 0.0075960
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0080392, 0.0084310
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0106418, 0.0101826
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0080527, 0.0076895
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089614, 0.0093082
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0080867, 0.0077076
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0295853, 0.0313576

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
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131806, upper bound: 0.0134179
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131767, upper bound: 0.0134160
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0078190, 0.0082682
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0056077, 0.0059054
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0072693, 0.0077401
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0081376, 0.0076580
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0096869, 0.0102550
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0073436, 0.0077910
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0090504, 0.0086545
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0073228, 0.0077948
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0300928, 0.0279330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 131

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0142856, upper bound: 0.0129133
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147121, upper bound: 0.0127658
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0079663, 0.0081223
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0057026, 0.0058106
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0073989, 0.0076198
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0079994, 0.0077983
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0098568, 0.0100872
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0074721, 0.0076685
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089444, 0.0087626
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0074612, 0.0076697
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0294756, 0.0285610

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0120009, upper bound: 0.0118881
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0120009, upper bound: 0.0118881
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0083409, 0.0085848
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058919, 0.0060553
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0069302, 0.0071372
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0075146, 0.0072892
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0094320, 0.0097037
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0069270, 0.0071316
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0084435, 0.0082561
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0073598, 0.0075813
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0285676, 0.0275562

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0130940, upper bound: 0.0127810
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0130940, upper bound: 0.0127810
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0083034, 0.0086442
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058669, 0.0060948
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0068168, 0.0072807
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0076306, 0.0072016
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0093291, 0.0098323
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0068305, 0.0072519
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0085459, 0.0081923
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0072565, 0.0077067
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0290988, 0.0271677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147281, upper bound: 0.0143091
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0150455, upper bound: 0.0143086
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0084155, 0.0087053
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059932, 0.0061866
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0077181, 0.0080694
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0084592, 0.0081186
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0102828, 0.0106829
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0077675, 0.0080929
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0092613, 0.0089723
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0078791, 0.0082267
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0318274, 0.0302884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129791, upper bound: 0.0128313
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129791, upper bound: 0.0128313
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0084020, 0.0087274
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059837, 0.0062024
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0077388, 0.0080686
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0084599, 0.0081254
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0102887, 0.0106831
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0077812, 0.0080924
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0092622, 0.0089774
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0078943, 0.0082260
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0318310, 0.0303184

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 131

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0150657, upper bound: 0.0139858
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138814, upper bound: 0.0149993
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0081814, 0.0084081
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058564, 0.0060083
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0073748, 0.0077190
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0081129, 0.0078055
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0098912, 0.0102623
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0074582, 0.0077728
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089443, 0.0087204
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0075231, 0.0078824
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0302860, 0.0289169

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132285, upper bound: 0.0127841
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132285, upper bound: 0.0127841
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0084241, 0.0084847
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059994, 0.0060593
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0077388, 0.0077253
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0081468, 0.0081254
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0102887, 0.0102915
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0077812, 0.0077831
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0090102, 0.0089774
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0078943, 0.0078700
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0304595, 0.0303184

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132285, upper bound: 0.0127841
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132285, upper bound: 0.0127841
time: 0.71 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.71 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0133271, upper bound: 0.0132904
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0133268, upper bound: 0.0132867
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0135580, upper bound: 0.0128587
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0135580, upper bound: 0.0128587
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0134260, upper bound: 0.0131022
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0134023, upper bound: 0.0130906
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0124476, upper bound: 0.0135623
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0126857, upper bound: 0.0131880
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0127160, upper bound: 0.0133331
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0127160, upper bound: 0.0133331
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0151858, upper bound: 0.0154000
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0151910, upper bound: 0.0151810
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0148301, upper bound: 0.0145404
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0149751, upper bound: 0.0143124
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0143368, upper bound: 0.0160553
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0143396, upper bound: 0.0158514
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0125642, upper bound: 0.0112900
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0125642, upper bound: 0.0112900
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0150528, upper bound: 0.0129617
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0151887, upper bound: 0.0129169
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0108162, upper bound: 0.0105294
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0108162, upper bound: 0.0105294
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0108033, upper bound: 0.0105294
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0108033, upper bound: 0.0105294
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0147851, upper bound: 0.0145243
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0147851, upper bound: 0.0145243
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0149598, upper bound: 0.0148402
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0150988, upper bound: 0.0148225
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0123295, upper bound: 0.0137055
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0123315, upper bound: 0.0136923
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0105294, upper bound: 0.0107950
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0105294, upper bound: 0.0108162
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0138358, upper bound: 0.0152980
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0137757, upper bound: 0.0156758
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0151619, upper bound: 0.0137790
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0155588, upper bound: 0.0136737
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0158851, upper bound: 0.0144547
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0160435, upper bound: 0.0144517
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0108474, upper bound: 0.0101706
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0108474, upper bound: 0.0102176
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0146045, upper bound: 0.0131310
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0147750, upper bound: 0.0131261
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0138476, upper bound: 0.0139623
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0135600, upper bound: 0.0139620
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0106502, upper bound: 0.0105709
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0106502, upper bound: 0.0105709
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0119352, upper bound: 0.0117163
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0115276, upper bound: 0.0122236
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0133391, upper bound: 0.0139349
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0133624, upper bound: 0.0137488
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0133790, upper bound: 0.0138234
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0133790, upper bound: 0.0138234
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0143555, upper bound: 0.0146378
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0130916, upper bound: 0.0151520
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0120445, upper bound: 0.0124058
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0121000, upper bound: 0.0121283
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0112607, upper bound: 0.0111165
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0107944, upper bound: 0.0114688
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0146370, upper bound: 0.0147654
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0146370, upper bound: 0.0147654
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0137550, upper bound: 0.0136482
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0132494, upper bound: 0.0136796
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0126666, upper bound: 0.0113386
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0126786, upper bound: 0.0113523
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0133557, upper bound: 0.0131356
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0133571, upper bound: 0.0131381
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0133439, upper bound: 0.0125896
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0128008, upper bound: 0.0130978
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0133712, upper bound: 0.0125187
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0128040, upper bound: 0.0129572
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0134815, upper bound: 0.0125501
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0128046, upper bound: 0.0129514
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0113487, upper bound: 0.0126786
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0113353, upper bound: 0.0126666
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0130991, upper bound: 0.0134167
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0130893, upper bound: 0.0134158
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0131806, upper bound: 0.0134179
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0131767, upper bound: 0.0134160
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0142856, upper bound: 0.0129133
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0147121, upper bound: 0.0127658
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0120009, upper bound: 0.0118881
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0120009, upper bound: 0.0118881
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0130940, upper bound: 0.0127810
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0130940, upper bound: 0.0127810
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0147281, upper bound: 0.0143091
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0150455, upper bound: 0.0143086
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0129791, upper bound: 0.0128313
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0129791, upper bound: 0.0128313
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0150657, upper bound: 0.0139858
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0138814, upper bound: 0.0149993
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0132285, upper bound: 0.0127841
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0132285, upper bound: 0.0127841
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0132285, upper bound: 0.0127841
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 9, lower bound: -0.0132285, upper bound: 0.0127841

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0083563, 0.0079901
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059972, 0.0057531
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0075889, 0.0071694
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0076308, 0.0080365
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0101503, 0.0096741
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0076703, 0.0072826
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0086130, 0.0089613
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0077001, 0.0072911
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0278817, 0.0297132

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0141397, upper bound: 0.0134803
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129866, upper bound: 0.0143104
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0082216, 0.0080530
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059095, 0.0057943
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0074394, 0.0072384
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0076952, 0.0078805
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0099661, 0.0097500
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0075269, 0.0073454
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0086715, 0.0088341
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0075516, 0.0073531
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0281706, 0.0290112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 131

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0120147, upper bound: 0.0120895
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0120147, upper bound: 0.0120903
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0086442, 0.0083034
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0060948, 0.0058669
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0072807, 0.0068168
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0072016, 0.0076306
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0098323, 0.0093291
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0072519, 0.0068305
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0081923, 0.0085459
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0077067, 0.0072565
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0271677, 0.0290988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139705, upper bound: 0.0137160
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139883, upper bound: 0.0135746
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0085848, 0.0083409
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0060553, 0.0058919
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0071372, 0.0069302
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0072892, 0.0075146
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0097037, 0.0094320
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0071316, 0.0069270
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0082561, 0.0084435
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0075813, 0.0073598
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0275562, 0.0285676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Candidate
type: RSZ, layer: 3, pos: 131

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0105214, upper bound: 0.0104121
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0105214, upper bound: 0.0104233
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0083563, 0.0079901
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059972, 0.0057531
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0075889, 0.0071694
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0076308, 0.0080365
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0101503, 0.0096741
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0076703, 0.0072826
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0086130, 0.0089613
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0077001, 0.0072911
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0278817, 0.0297132

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0105761, upper bound: 0.0116513
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0105322, upper bound: 0.0116513
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0082216, 0.0080530
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059095, 0.0057943
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0074394, 0.0072384
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0076952, 0.0078805
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0099661, 0.0097500
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0075269, 0.0073454
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0086715, 0.0088341
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0075516, 0.0073531
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0281706, 0.0290112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0128667, upper bound: 0.0144734
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129982, upper bound: 0.0141051
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0079948, 0.0082888
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0057517, 0.0059582
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0072100, 0.0074726
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0079201, 0.0076622
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0097130, 0.0100118
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0073177, 0.0075603
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0088772, 0.0086359
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0073317, 0.0075808
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0291926, 0.0280206

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 131

### Candidate
type: RSZ, layer: 3, pos: 180

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0135719, upper bound: 0.0117673
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138147, upper bound: 0.0115867
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0079274, 0.0084127
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0057084, 0.0060390
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0071337, 0.0076152
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0080652, 0.0075869
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0096236, 0.0101825
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0072464, 0.0076956
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089945, 0.0085711
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0072611, 0.0077200
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0298452, 0.0276826

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 180

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 131

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0137012, upper bound: 0.0117141
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139669, upper bound: 0.0114892
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0083537, 0.0087617
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059479, 0.0062284
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0076898, 0.0080938
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0084873, 0.0080856
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0102458, 0.0107143
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0077398, 0.0081163
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0092938, 0.0089368
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0078576, 0.0082442
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0319574, 0.0301383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 131

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138562, upper bound: 0.0136755
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139671, upper bound: 0.0136592
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0083438, 0.0087838
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059411, 0.0062441
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0077104, 0.0080949
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0084886, 0.0080924
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0102518, 0.0107153
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0077536, 0.0081178
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0092954, 0.0089418
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0078728, 0.0082460
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0319630, 0.0301684

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 131

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138562, upper bound: 0.0136755
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139671, upper bound: 0.0136592
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0079948, 0.0082888
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0057517, 0.0059582
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0072100, 0.0074726
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0079201, 0.0076622
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0097130, 0.0100118
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0073177, 0.0075603
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0088772, 0.0086359
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0073317, 0.0075808
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0291926, 0.0280206

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132882, upper bound: 0.0136747
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132916, upper bound: 0.0136747
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0079274, 0.0084127
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0057084, 0.0060390
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0071337, 0.0076152
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0080652, 0.0075869
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0096236, 0.0101825
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0072464, 0.0076956
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089945, 0.0085711
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0072611, 0.0077200
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0298452, 0.0276826

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0119431, upper bound: 0.0116359
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0117760, upper bound: 0.0120500
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0086157, 0.0082599
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0061321, 0.0058871
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0079063, 0.0075967
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0080473, 0.0083614
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0105597, 0.0101946
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0079821, 0.0076930
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089627, 0.0092540
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0080098, 0.0077148
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0296176, 0.0310465

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0113081, upper bound: 0.0126324
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0112930, upper bound: 0.0126209
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0086157, 0.0082599
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0061321, 0.0058871
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0079063, 0.0075967
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0080473, 0.0083614
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0105597, 0.0101946
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0079821, 0.0076930
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089627, 0.0092540
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0080098, 0.0077148
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0296176, 0.0310465

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0100472, upper bound: 0.0107653
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0100139, upper bound: 0.0107653
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0086501, 0.0082691
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0061546, 0.0058954
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0079635, 0.0076163
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0080616, 0.0084151
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0106237, 0.0102079
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0080358, 0.0077076
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089780, 0.0092942
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0080713, 0.0077285
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0296884, 0.0312873

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129553, upper bound: 0.0148513
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129558, upper bound: 0.0146506
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0083289, 0.0085994
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058836, 0.0060658
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0069107, 0.0071473
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0075242, 0.0072724
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0094122, 0.0097142
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0069092, 0.0071403
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0084563, 0.0082444
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0073435, 0.0075896
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0286129, 0.0274841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Candidate
type: RSZ, layer: 3, pos: 241

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0141485, upper bound: 0.0129650
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0143277, upper bound: 0.0129629
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0082917, 0.0086608
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058587, 0.0061069
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0068031, 0.0072987
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0076468, 0.0071904
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0093161, 0.0098508
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0068179, 0.0072687
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0085600, 0.0081809
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0072402, 0.0077222
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0291714, 0.0271156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Candidate
type: RSZ, layer: 3, pos: 131

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0145610, upper bound: 0.0128582
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147313, upper bound: 0.0128489
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0080411, 0.0082373
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0057861, 0.0059212
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0072189, 0.0074582
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0078992, 0.0076783
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0097303, 0.0099885
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0073275, 0.0075447
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0088512, 0.0086598
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0073368, 0.0075691
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0290935, 0.0280986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 241

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0141109, upper bound: 0.0131351
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0144887, upper bound: 0.0130909
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0079781, 0.0083730
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0057454, 0.0060093
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0071501, 0.0076069
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0080527, 0.0076101
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0096495, 0.0101688
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0072632, 0.0076871
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089754, 0.0086002
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0072739, 0.0077155
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0297858, 0.0277904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0142406, upper bound: 0.0131346
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0146563, upper bound: 0.0130858
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0080411, 0.0082373
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0057861, 0.0059212
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0072189, 0.0074582
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0078992, 0.0076783
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0097303, 0.0099885
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0073275, 0.0075447
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0088512, 0.0086598
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0073368, 0.0075691
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0290935, 0.0280986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0100650, upper bound: 0.0093758
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0100650, upper bound: 0.0094251
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0079781, 0.0083730
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0057454, 0.0060093
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0071501, 0.0076069
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0080527, 0.0076101
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0096495, 0.0101688
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0072632, 0.0076871
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089754, 0.0086002
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0072739, 0.0077155
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0297858, 0.0277904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0102934, upper bound: 0.0092682
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0102934, upper bound: 0.0093698
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0082921, 0.0086259
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059128, 0.0061361
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0075978, 0.0079737
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0084185, 0.0080433
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0101861, 0.0106282
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0076900, 0.0080442
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0092891, 0.0089752
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0077118, 0.0080822
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0312982, 0.0296075

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 180

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0130112, upper bound: 0.0131518
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0130498, upper bound: 0.0131513
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0082940, 0.0086016
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059129, 0.0061201
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0075857, 0.0079243
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0083748, 0.0080441
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0101897, 0.0105765
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0076846, 0.0079984
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0092582, 0.0089735
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0077035, 0.0080252
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0311039, 0.0296109

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 180

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0101402, upper bound: 0.0105841
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0101402, upper bound: 0.0105841
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0083730, 0.0079781
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0060093, 0.0057454
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0076069, 0.0071501
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0076101, 0.0080527
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0101688, 0.0096495
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0076871, 0.0072632
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0086002, 0.0089754
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0077155, 0.0072739
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0277904, 0.0297858

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 241

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0111196, upper bound: 0.0112956
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0105729, upper bound: 0.0116513
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0082373, 0.0080411
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059212, 0.0057861
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0074582, 0.0072189
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0076783, 0.0078992
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0099885, 0.0097303
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0075447, 0.0073275
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0086598, 0.0088512
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0075691, 0.0073368
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0280986, 0.0290935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0119421, upper bound: 0.0123701
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0120146, upper bound: 0.0120925
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0087261, 0.0083901
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0062013, 0.0059755
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0080867, 0.0076877
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0080981, 0.0084761
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0107016, 0.0102598
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0081092, 0.0077421
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089581, 0.0092763
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0082415, 0.0078523
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0301990, 0.0319036

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0111732, upper bound: 0.0109926
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0107041, upper bound: 0.0115227
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0087220, 0.0084122
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0061987, 0.0059912
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0081073, 0.0076986
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0081017, 0.0084830
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0107076, 0.0102631
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0081230, 0.0077497
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089606, 0.0092813
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0082566, 0.0078628
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0302164, 0.0319336

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 131

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 43

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0119617, upper bound: 0.0124630
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0120301, upper bound: 0.0122036
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0081389, 0.0079511
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058227, 0.0056924
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0076379, 0.0073690
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0077767, 0.0080156
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0101057, 0.0098297
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0076854, 0.0074445
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0087504, 0.0089584
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0076851, 0.0074304
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0284678, 0.0295482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 43

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0130474, upper bound: 0.0133805
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131513, upper bound: 0.0130498
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0082911, 0.0078070
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059214, 0.0055995
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0077624, 0.0072498
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0076412, 0.0081570
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0102781, 0.0096672
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0078094, 0.0073258
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0086428, 0.0090694
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0078148, 0.0073064
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0278610, 0.0301813

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 131

### Candidate
type: RSZ, layer: 3, pos: 43

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117420, upper bound: 0.0138940
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0119027, upper bound: 0.0134783
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0083650, 0.0087462
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059564, 0.0062172
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0077096, 0.0080748
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0084713, 0.0081049
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0102681, 0.0106962
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0077580, 0.0080998
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0092799, 0.0089499
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0078741, 0.0082293
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0318839, 0.0302230

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 131

### Candidate
type: RSZ, layer: 3, pos: 180

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0107754, upper bound: 0.0104874
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0107546, upper bound: 0.0104874
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0083560, 0.0087683
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059497, 0.0062330
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0077303, 0.0080764
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0084727, 0.0081118
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0102740, 0.0106971
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0077718, 0.0081009
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0092814, 0.0089550
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0078893, 0.0082306
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0318928, 0.0302531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0107754, upper bound: 0.0104874
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0107546, upper bound: 0.0104874
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0081353, 0.0084484
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058224, 0.0060389
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0073663, 0.0077279
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0081258, 0.0077918
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0098764, 0.0102763
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0074487, 0.0077823
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089634, 0.0086980
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0075181, 0.0078879
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0303439, 0.0288515

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129214, upper bound: 0.0128442
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129445, upper bound: 0.0128439
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0083409, 0.0085848
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058919, 0.0060553
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0069302, 0.0071372
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0075146, 0.0072892
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0094320, 0.0097037
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0069270, 0.0071316
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0084435, 0.0082561
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0073598, 0.0075813
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0285676, 0.0275562

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 131

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133562, upper bound: 0.0120713
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0134783, upper bound: 0.0120156
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0083034, 0.0086442
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058669, 0.0060948
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0068168, 0.0072807
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0076306, 0.0072016
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0093291, 0.0098323
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0068305, 0.0072519
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0085459, 0.0081923
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0072565, 0.0077067
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0290988, 0.0271677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 131

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0136975, upper bound: 0.0119130
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138917, upper bound: 0.0118333
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0080531, 0.0082216
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0057943, 0.0059095
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0072384, 0.0074394
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0078805, 0.0076952
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0097500, 0.0099661
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0073454, 0.0075269
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0088341, 0.0086715
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0073531, 0.0075516
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0290112, 0.0281706

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
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 131

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0135985, upper bound: 0.0125574
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0124086, upper bound: 0.0130846
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0079901, 0.0083563
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0057531, 0.0059972
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0071694, 0.0075889
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0080365, 0.0076308
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0096741, 0.0101503
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0072826, 0.0076703
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089613, 0.0086130
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0072911, 0.0077001
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0297133, 0.0278817

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138789, upper bound: 0.0125576
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128173, upper bound: 0.0130836
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0078190, 0.0082682
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0056077, 0.0059054
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0072693, 0.0077401
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0081376, 0.0076580
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0096869, 0.0102550
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0073436, 0.0077910
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0090504, 0.0086545
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0073228, 0.0077948
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0300928, 0.0279330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 241

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0106849, upper bound: 0.0101462
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0106849, upper bound: 0.0101816
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0079663, 0.0081223
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0057026, 0.0058106
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0073989, 0.0076198
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0079994, 0.0077983
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0098568, 0.0100872
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0074721, 0.0076685
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089444, 0.0087626
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0074612, 0.0076697
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0294756, 0.0285610

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0101397, upper bound: 0.0105841
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0101188, upper bound: 0.0105841
time: 0.72 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 4.59 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0141397, upper bound: 0.0134803
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0129866, upper bound: 0.0143104
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0120147, upper bound: 0.0120895
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0120147, upper bound: 0.0120903
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0139705, upper bound: 0.0137160
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0139883, upper bound: 0.0135746
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0105214, upper bound: 0.0104121
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0105214, upper bound: 0.0104233
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0105761, upper bound: 0.0116513
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0105322, upper bound: 0.0116513
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0128667, upper bound: 0.0144734
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0129982, upper bound: 0.0141051
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0135719, upper bound: 0.0117673
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0138147, upper bound: 0.0115867
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0137012, upper bound: 0.0117141
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0139669, upper bound: 0.0114892
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0138562, upper bound: 0.0136755
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0139671, upper bound: 0.0136592
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0138562, upper bound: 0.0136755
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0139671, upper bound: 0.0136592
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0132882, upper bound: 0.0136747
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0132916, upper bound: 0.0136747
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0119431, upper bound: 0.0116359
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0117760, upper bound: 0.0120500
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0113081, upper bound: 0.0126324
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0112930, upper bound: 0.0126209
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0100472, upper bound: 0.0107653
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0100139, upper bound: 0.0107653
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0129553, upper bound: 0.0148513
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0129558, upper bound: 0.0146506
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0141485, upper bound: 0.0129650
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0143277, upper bound: 0.0129629
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0145610, upper bound: 0.0128582
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0147313, upper bound: 0.0128489
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0141109, upper bound: 0.0131351
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0144887, upper bound: 0.0130909
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0142406, upper bound: 0.0131346
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0146563, upper bound: 0.0130858
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0100650, upper bound: 0.0093758
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0100650, upper bound: 0.0094251
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0102934, upper bound: 0.0092682
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0102934, upper bound: 0.0093698
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0130112, upper bound: 0.0131518
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0130498, upper bound: 0.0131513
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0101402, upper bound: 0.0105841
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0101402, upper bound: 0.0105841
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0111196, upper bound: 0.0112956
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0105729, upper bound: 0.0116513
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0119421, upper bound: 0.0123701
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0120146, upper bound: 0.0120925
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0111732, upper bound: 0.0109926
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0107041, upper bound: 0.0115227
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0119617, upper bound: 0.0124630
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0120301, upper bound: 0.0122036
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0130474, upper bound: 0.0133805
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0131513, upper bound: 0.0130498
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0117420, upper bound: 0.0138940
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0119027, upper bound: 0.0134783
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0107754, upper bound: 0.0104874
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0107546, upper bound: 0.0104874
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0107754, upper bound: 0.0104874
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0107546, upper bound: 0.0104874
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0129214, upper bound: 0.0128442
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0129445, upper bound: 0.0128439
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0133562, upper bound: 0.0120713
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0134783, upper bound: 0.0120156
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0136975, upper bound: 0.0119130
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0138917, upper bound: 0.0118333
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0135985, upper bound: 0.0125574
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0124086, upper bound: 0.0130846
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0138789, upper bound: 0.0125576
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0128173, upper bound: 0.0130836
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0106849, upper bound: 0.0101462
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0106849, upper bound: 0.0101816
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0101397, upper bound: 0.0105841
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 9, lower bound: -0.0101188, upper bound: 0.0105841

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0081223, 0.0079663
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058106, 0.0057026
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0076198, 0.0073989
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0077983, 0.0079994
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0100872, 0.0098568
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0076685, 0.0074721
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0087626, 0.0089444
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0076697, 0.0074612
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0285611, 0.0294756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0097469, upper bound: 0.0096787
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0097469, upper bound: 0.0096964
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0082682, 0.0078190
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059054, 0.0056077
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0077401, 0.0072693
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0076580, 0.0081376
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0102550, 0.0096869
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0077910, 0.0073436
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0086545, 0.0090504
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0077948, 0.0073228
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0279330, 0.0300928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0091920, upper bound: 0.0100895
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0091284, upper bound: 0.0100895
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0083563, 0.0079901
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059972, 0.0057531
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0075889, 0.0071694
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0076308, 0.0080365
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0101503, 0.0096741
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0076703, 0.0072826
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0086130, 0.0089613
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0077001, 0.0072911
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0278817, 0.0297132

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 131

### Candidate
type: RSZ, layer: 3, pos: 241

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0096343, upper bound: 0.0098842
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0096343, upper bound: 0.0098879
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0082216, 0.0080530
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059095, 0.0057943
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0074394, 0.0072384
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0076952, 0.0078805
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0099661, 0.0097500
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0075269, 0.0073454
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0086715, 0.0088341
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0075516, 0.0073531
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0281706, 0.0290112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0096932, upper bound: 0.0095875
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0096932, upper bound: 0.0096482
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0086442, 0.0083034
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0060948, 0.0058669
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0072807, 0.0068168
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0072016, 0.0076306
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0098323, 0.0093291
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0072519, 0.0068305
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0081923, 0.0085459
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0077067, 0.0072565
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0271677, 0.0290988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 131

### Candidate
type: RSZ, layer: 3, pos: 43

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0091503, upper bound: 0.0100254
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0090809, upper bound: 0.0100254
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0085848, 0.0083409
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0060553, 0.0058919
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0071372, 0.0069302
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0072892, 0.0075146
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0097037, 0.0094320
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0071316, 0.0069270
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0082561, 0.0084435
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0075813, 0.0073598
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0275562, 0.0285676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0092524, upper bound: 0.0098517
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0092208, upper bound: 0.0098517
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0082294, 0.0087005
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058137, 0.0061366
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0067579, 0.0073070
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0076593, 0.0071438
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0092650, 0.0098644
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0067763, 0.0072772
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0085792, 0.0081332
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0072042, 0.0077267
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0292308, 0.0269018

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Candidate
type: RSZ, layer: 3, pos: 43

### Candidate
type: RSZ, layer: 3, pos: 131

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0100136, upper bound: 0.0091197
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0100136, upper bound: 0.0091197
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0082294, 0.0087005
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058137, 0.0061366
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0067579, 0.0073070
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0076593, 0.0071438
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0092650, 0.0098644
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0067763, 0.0072772
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0085792, 0.0081332
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0072042, 0.0077267
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0292308, 0.0269018

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0101798, upper bound: 0.0089885
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0101798, upper bound: 0.0089885
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0079948, 0.0082888
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0057517, 0.0059582
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0072100, 0.0074726
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0079201, 0.0076622
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0097130, 0.0100118
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0073177, 0.0075603
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0088772, 0.0086359
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0073317, 0.0075808
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0291926, 0.0280206

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 241

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0098821, upper bound: 0.0096090
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0098679, upper bound: 0.0096090
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0079274, 0.0084127
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0057084, 0.0060390
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0071337, 0.0076152
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0080652, 0.0075869
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0096236, 0.0101825
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0072464, 0.0076956
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089945, 0.0085711
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0072611, 0.0077200
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0298452, 0.0276826

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 131

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 180

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 241

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0101369, upper bound: 0.0095036
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0101229, upper bound: 0.0095036
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0079948, 0.0082888
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0057517, 0.0059582
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0072100, 0.0074726
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0079201, 0.0076622
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0097130, 0.0100118
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0073177, 0.0075603
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0088772, 0.0086359
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0073317, 0.0075808
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0291926, 0.0280206

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0098821, upper bound: 0.0096090
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0098679, upper bound: 0.0096090
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0079274, 0.0084127
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0057084, 0.0060390
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0071337, 0.0076152
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0080652, 0.0075869
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0096236, 0.0101825
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0072464, 0.0076956
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089945, 0.0085711
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0072611, 0.0077200
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0298452, 0.0276826

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 131

### Candidate
type: RSZ, layer: 3, pos: 241

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0101369, upper bound: 0.0095036
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0101229, upper bound: 0.0095036
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0083972, 0.0079415
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0060278, 0.0057182
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0075967, 0.0071539
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0076109, 0.0080493
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0101643, 0.0096517
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0076788, 0.0072666
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0085845, 0.0089805
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0077046, 0.0072804
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0277883, 0.0297750

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 180

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0091651, upper bound: 0.0104087
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0090982, upper bound: 0.0104087
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0082722, 0.0080070
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0059465, 0.0057603
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0074535, 0.0072299
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0076815, 0.0079027
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0099913, 0.0097353
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0075421, 0.0073359
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0086490, 0.0088600
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0075638, 0.0073482
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0281053, 0.0291133

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Candidate
type: RSZ, layer: 3, pos: 131

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0092125, upper bound: 0.0101582
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0091816, upper bound: 0.0101582
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0080411, 0.0082373
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0057861, 0.0059212
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0072189, 0.0074582
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0078992, 0.0076783
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0097303, 0.0099885
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0073275, 0.0075447
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0088512, 0.0086598
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0073368, 0.0075691
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0290935, 0.0280986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0098520, upper bound: 0.0091996
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0098520, upper bound: 0.0092520
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0079781, 0.0083730
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0057454, 0.0060093
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0071501, 0.0076069
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0080527, 0.0076101
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0096495, 0.0101688
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0072632, 0.0076871
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089754, 0.0086002
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0072739, 0.0077155
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0297858, 0.0277904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Candidate
type: RSZ, layer: 3, pos: 180

### Candidate
type: RSZ, layer: 3, pos: 131

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0100895, upper bound: 0.0090867
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0100895, upper bound: 0.0091900
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0080411, 0.0082373
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0057861, 0.0059212
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0072189, 0.0074582
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0078992, 0.0076783
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0097303, 0.0099885
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0073275, 0.0075447
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0088512, 0.0086598
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0073368, 0.0075691
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0290935, 0.0280986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 131

### Candidate
type: RSZ, layer: 3, pos: 241

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0100273, upper bound: 0.0090753
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0100273, upper bound: 0.0091503
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0079781, 0.0083730
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0057454, 0.0060093
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0071501, 0.0076069
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0080527, 0.0076101
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0096495, 0.0101688
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0072632, 0.0076871
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089754, 0.0086002
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0072739, 0.0077155
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0297858, 0.0277904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 131

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0102549, upper bound: 0.0089401
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0102549, upper bound: 0.0090698
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0083289, 0.0085994
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058836, 0.0060658
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0069107, 0.0071473
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0075242, 0.0072724
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0094122, 0.0097142
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0069092, 0.0071403
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0084563, 0.0082444
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0073435, 0.0075896
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0286129, 0.0274841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Candidate
type: RSZ, layer: 3, pos: 180

### Candidate
type: RSZ, layer: 3, pos: 131

### Candidate
type: RSZ, layer: 3, pos: 43

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0097666, upper bound: 0.0093769
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0097666, upper bound: 0.0094407
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0082917, 0.0086608
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058587, 0.0061069
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0068031, 0.0072987
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0076468, 0.0071904
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0093161, 0.0098508
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0068179, 0.0072687
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0085600, 0.0081809
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0072402, 0.0077222
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0291714, 0.0271156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Candidate
type: RSZ, layer: 3, pos: 131

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0099458, upper bound: 0.0093386
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0099458, upper bound: 0.0093880
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0083289, 0.0085994
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058836, 0.0060658
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0069107, 0.0071473
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0075242, 0.0072724
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0094122, 0.0097142
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0069092, 0.0071403
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0084563, 0.0082444
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0073435, 0.0075896
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0286129, 0.0274841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 131

### Candidate
type: RSZ, layer: 3, pos: 180

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0099601, upper bound: 0.0093148
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0099601, upper bound: 0.0094043
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0082917, 0.0086608
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0058587, 0.0061069
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0068031, 0.0072987
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0076468, 0.0071904
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0093161, 0.0098508
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0068179, 0.0072687
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0085600, 0.0081809
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0072402, 0.0077222
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0291714, 0.0271156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 180

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0101387, upper bound: 0.0092305
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0101387, upper bound: 0.0093332
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0086608, 0.0082917
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0061069, 0.0058587
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0072987, 0.0068031
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0071904, 0.0076468
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0098508, 0.0093161
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0072687, 0.0068179
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0081809, 0.0085600
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0077222, 0.0072402
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0271156, 0.0291714

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 131

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0089401, upper bound: 0.0102549
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0089401, upper bound: 0.0102549
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0079901, 0.0083563
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0057531, 0.0059972
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0071694, 0.0075889
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0080365, 0.0076308
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0096741, 0.0101503
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0072826, 0.0076703
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0089613, 0.0086130
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0072911, 0.0077001
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0297133, 0.0278817

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 131

### Candidate
type: RSZ, layer: 3, pos: 180

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0102549, upper bound: 0.0089611
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0102549, upper bound: 0.0089611
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040962, -0.0010808, -0.0040962, -0.0010808, -0.0030154, 0.0030154
1: 0.0178105, 0.0325450, 0.0178105, 0.0325450, -0.0078190, 0.0082682
2: 0.0204607, 0.0305060, 0.0204607, 0.0305060, -0.0056077, 0.0059054
3: 0.0061783, 0.0176508, 0.0061783, 0.0176508, -0.0072693, 0.0077401
4: -0.0185718, -0.0072829, -0.0185718, -0.0072829, -0.0081376, 0.0076580
5: 0.0125542, 0.0267461, 0.0125542, 0.0267461, -0.0096869, 0.0102550
6: 0.0048297, 0.0155457, 0.0048297, 0.0155457, -0.0073436, 0.0077910
7: -0.0231210, -0.0114719, -0.0231210, -0.0114719, -0.0090504, 0.0086545
8: 0.0077075, 0.0192184, 0.0077075, 0.0192184, -0.0073228, 0.0077948
9: 0.8984746, 0.9487647, 0.8984746, 0.9487647, -0.0300928, 0.0279330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0102549, upper bound: 0.0090698
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0102549, upper bound: 0.0090698
time: 0.74 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 2.89 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0097469, upper bound: 0.0096787
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0097469, upper bound: 0.0096964
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0091920, upper bound: 0.0100895
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0091284, upper bound: 0.0100895
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0096343, upper bound: 0.0098842
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0096343, upper bound: 0.0098879
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0096932, upper bound: 0.0095875
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0096932, upper bound: 0.0096482
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0091503, upper bound: 0.0100254
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0090809, upper bound: 0.0100254
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0092524, upper bound: 0.0098517
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0092208, upper bound: 0.0098517
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0100136, upper bound: 0.0091197
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0100136, upper bound: 0.0091197
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0101798, upper bound: 0.0089885
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0101798, upper bound: 0.0089885
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0098821, upper bound: 0.0096090
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0098679, upper bound: 0.0096090
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0101369, upper bound: 0.0095036
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0101229, upper bound: 0.0095036
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0098821, upper bound: 0.0096090
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0098679, upper bound: 0.0096090
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0101369, upper bound: 0.0095036
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0101229, upper bound: 0.0095036
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0091651, upper bound: 0.0104087
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0090982, upper bound: 0.0104087
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0092125, upper bound: 0.0101582
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0091816, upper bound: 0.0101582
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0098520, upper bound: 0.0091996
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0098520, upper bound: 0.0092520
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0100895, upper bound: 0.0090867
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0100895, upper bound: 0.0091900
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0100273, upper bound: 0.0090753
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0100273, upper bound: 0.0091503
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0102549, upper bound: 0.0089401
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0102549, upper bound: 0.0090698
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0097666, upper bound: 0.0093769
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0097666, upper bound: 0.0094407
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0099458, upper bound: 0.0093386
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0099458, upper bound: 0.0093880
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0099601, upper bound: 0.0093148
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0099601, upper bound: 0.0094043
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0101387, upper bound: 0.0092305
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0101387, upper bound: 0.0093332
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0089401, upper bound: 0.0102549
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0089401, upper bound: 0.0102549
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0102549, upper bound: 0.0089611
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0102549, upper bound: 0.0089611
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0102549, upper bound: 0.0090698
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.89
Output dim: 9, lower bound: -0.0102549, upper bound: 0.0090698

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.90 + 569.74 = 572.63 seconds
