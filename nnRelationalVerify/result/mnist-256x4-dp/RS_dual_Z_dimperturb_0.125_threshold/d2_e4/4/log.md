## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00058656


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0036913, 0.0046427, 0.0036913, 0.0046427, -0.0007131, 0.0007131)
1: (0.0018556, 0.0019930, 0.0018556, 0.0019930, -0.0001030, 0.0001030)
2: (0.0117930, 0.0123190, 0.0117930, 0.0123190, -0.0003943, 0.0003943)
3: (-0.0024836, -0.0019395, -0.0024836, -0.0019395, -0.0004078, 0.0004078)
4: (-0.0019373, -0.0013483, -0.0019373, -0.0013483, -0.0004414, 0.0004414)
5: (0.0053893, 0.0059467, 0.0053893, 0.0059467, -0.0004178, 0.0004178)
6: (-0.0009170, 0.0012943, -0.0009170, 0.0012943, -0.0016575, 0.0016575)
7: (-0.0043195, -0.0013078, -0.0043195, -0.0013078, -0.0022574, 0.0022574)
8: (0.9861712, 0.9882926, 0.9861712, 0.9882926, -0.0015902, 0.0015902)
9: (-0.0052601, -0.0033343, -0.0052601, -0.0033343, -0.0014435, 0.0014435)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.59 + 1.31 = 2.90 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0006966, upper bound: 0.0006967

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006741, upper bound: 0.0006883
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006883, upper bound: 0.0006742
time: 0.47 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.16 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.16
Output dim: 8, lower bound: -0.0006741, upper bound: 0.0006883
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.16
Output dim: 8, lower bound: -0.0006883, upper bound: 0.0006742

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0036913, 0.0046427, 0.0036913, 0.0046427, -0.0007048, 0.0007055
1: 0.0018556, 0.0019930, 0.0018556, 0.0019930, -0.0001018, 0.0001019
2: 0.0117930, 0.0123190, 0.0117930, 0.0123190, -0.0003901, 0.0003897
3: -0.0024836, -0.0019395, -0.0024836, -0.0019395, -0.0004034, 0.0004030
4: -0.0019373, -0.0013483, -0.0019373, -0.0013483, -0.0004363, 0.0004367
5: 0.0053893, 0.0059467, 0.0053893, 0.0059467, -0.0004133, 0.0004129
6: -0.0009170, 0.0012943, -0.0009170, 0.0012943, -0.0016398, 0.0016382
7: -0.0043195, -0.0013078, -0.0043195, -0.0013078, -0.0022311, 0.0022332
8: 0.9861712, 0.9882926, 0.9861712, 0.9882926, -0.0015717, 0.0015731
9: -0.0052601, -0.0033343, -0.0052601, -0.0033343, -0.0014280, 0.0014267

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006348, upper bound: 0.0006623
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006447, upper bound: 0.0006451
time: 0.47 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0036913, 0.0046427, 0.0036913, 0.0046427, -0.0007055, 0.0007048
1: 0.0018556, 0.0019930, 0.0018556, 0.0019930, -0.0001019, 0.0001018
2: 0.0117930, 0.0123190, 0.0117930, 0.0123190, -0.0003897, 0.0003901
3: -0.0024836, -0.0019395, -0.0024836, -0.0019395, -0.0004030, 0.0004034
4: -0.0019373, -0.0013483, -0.0019373, -0.0013483, -0.0004367, 0.0004363
5: 0.0053893, 0.0059467, 0.0053893, 0.0059467, -0.0004129, 0.0004133
6: -0.0009170, 0.0012943, -0.0009170, 0.0012943, -0.0016382, 0.0016398
7: -0.0043195, -0.0013078, -0.0043195, -0.0013078, -0.0022332, 0.0022311
8: 0.9861712, 0.9882926, 0.9861712, 0.9882926, -0.0015731, 0.0015717
9: -0.0052601, -0.0033343, -0.0052601, -0.0033343, -0.0014267, 0.0014280

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006451, upper bound: 0.0006447
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006623, upper bound: 0.0006348
time: 0.47 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.57 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.57
Output dim: 8, lower bound: -0.0006348, upper bound: 0.0006623
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.57
Output dim: 8, lower bound: -0.0006447, upper bound: 0.0006451
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.57
Output dim: 8, lower bound: -0.0006451, upper bound: 0.0006447
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.57
Output dim: 8, lower bound: -0.0006623, upper bound: 0.0006348

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0036913, 0.0046427, 0.0036913, 0.0046427, -0.0006002, 0.0006083
1: 0.0018556, 0.0019930, 0.0018556, 0.0019930, -0.0000867, 0.0000879
2: 0.0117930, 0.0123190, 0.0117930, 0.0123190, -0.0003363, 0.0003318
3: -0.0024836, -0.0019395, -0.0024836, -0.0019395, -0.0003478, 0.0003432
4: -0.0019373, -0.0013483, -0.0019373, -0.0013483, -0.0003715, 0.0003765
5: 0.0053893, 0.0059467, 0.0053893, 0.0059467, -0.0003563, 0.0003516
6: -0.0009170, 0.0012943, -0.0009170, 0.0012943, -0.0014138, 0.0013950
7: -0.0043195, -0.0013078, -0.0043195, -0.0013078, -0.0018998, 0.0019255
8: 0.9861712, 0.9882926, 0.9861712, 0.9882926, -0.0013383, 0.0013564
9: -0.0052601, -0.0033343, -0.0052601, -0.0033343, -0.0012312, 0.0012148

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006061, upper bound: 0.0006298
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006053, upper bound: 0.0006298
time: 0.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0036913, 0.0046427, 0.0036913, 0.0046427, -0.0006059, 0.0006008
1: 0.0018556, 0.0019930, 0.0018556, 0.0019930, -0.0000875, 0.0000868
2: 0.0117930, 0.0123190, 0.0117930, 0.0123190, -0.0003322, 0.0003350
3: -0.0024836, -0.0019395, -0.0024836, -0.0019395, -0.0003436, 0.0003464
4: -0.0019373, -0.0013483, -0.0019373, -0.0013483, -0.0003750, 0.0003719
5: 0.0053893, 0.0059467, 0.0053893, 0.0059467, -0.0003520, 0.0003549
6: -0.0009170, 0.0012943, -0.0009170, 0.0012943, -0.0013965, 0.0014082
7: -0.0043195, -0.0013078, -0.0043195, -0.0013078, -0.0019178, 0.0019019
8: 0.9861712, 0.9882926, 0.9861712, 0.9882926, -0.0013510, 0.0013397
9: -0.0052601, -0.0033343, -0.0052601, -0.0033343, -0.0012161, 0.0012263

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006162, upper bound: 0.0006156
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006158, upper bound: 0.0006156
time: 0.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0036913, 0.0046427, 0.0036913, 0.0046427, -0.0006008, 0.0006059
1: 0.0018556, 0.0019930, 0.0018556, 0.0019930, -0.0000868, 0.0000875
2: 0.0117930, 0.0123190, 0.0117930, 0.0123190, -0.0003350, 0.0003322
3: -0.0024836, -0.0019395, -0.0024836, -0.0019395, -0.0003464, 0.0003436
4: -0.0019373, -0.0013483, -0.0019373, -0.0013483, -0.0003719, 0.0003750
5: 0.0053893, 0.0059467, 0.0053893, 0.0059467, -0.0003549, 0.0003520
6: -0.0009170, 0.0012943, -0.0009170, 0.0012943, -0.0014082, 0.0013965
7: -0.0043195, -0.0013078, -0.0043195, -0.0013078, -0.0019019, 0.0019178
8: 0.9861712, 0.9882926, 0.9861712, 0.9882926, -0.0013397, 0.0013510
9: -0.0052601, -0.0033343, -0.0052601, -0.0033343, -0.0012263, 0.0012161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006156, upper bound: 0.0006158
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006156, upper bound: 0.0006162
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0036913, 0.0046427, 0.0036913, 0.0046427, -0.0006083, 0.0006002
1: 0.0018556, 0.0019930, 0.0018556, 0.0019930, -0.0000879, 0.0000867
2: 0.0117930, 0.0123190, 0.0117930, 0.0123190, -0.0003318, 0.0003363
3: -0.0024836, -0.0019395, -0.0024836, -0.0019395, -0.0003432, 0.0003478
4: -0.0019373, -0.0013483, -0.0019373, -0.0013483, -0.0003765, 0.0003715
5: 0.0053893, 0.0059467, 0.0053893, 0.0059467, -0.0003516, 0.0003563
6: -0.0009170, 0.0012943, -0.0009170, 0.0012943, -0.0013950, 0.0014138
7: -0.0043195, -0.0013078, -0.0043195, -0.0013078, -0.0019255, 0.0018998
8: 0.9861712, 0.9882926, 0.9861712, 0.9882926, -0.0013564, 0.0013383
9: -0.0052601, -0.0033343, -0.0052601, -0.0033343, -0.0012148, 0.0012312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006298, upper bound: 0.0006053
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006298, upper bound: 0.0006061
time: 0.50 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.71 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.71
Output dim: 8, lower bound: -0.0006061, upper bound: 0.0006298
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.71
Output dim: 8, lower bound: -0.0006053, upper bound: 0.0006298
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.71
Output dim: 8, lower bound: -0.0006162, upper bound: 0.0006156
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.71
Output dim: 8, lower bound: -0.0006158, upper bound: 0.0006156
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.71
Output dim: 8, lower bound: -0.0006156, upper bound: 0.0006158
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.71
Output dim: 8, lower bound: -0.0006156, upper bound: 0.0006162
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.71
Output dim: 8, lower bound: -0.0006298, upper bound: 0.0006053
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.71
Output dim: 8, lower bound: -0.0006298, upper bound: 0.0006061

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0036913, 0.0046427, 0.0036913, 0.0046427, -0.0005897, 0.0005944
1: 0.0018556, 0.0019930, 0.0018556, 0.0019930, -0.0000852, 0.0000859
2: 0.0117930, 0.0123190, 0.0117930, 0.0123190, -0.0003286, 0.0003260
3: -0.0024836, -0.0019395, -0.0024836, -0.0019395, -0.0003399, 0.0003372
4: -0.0019373, -0.0013483, -0.0019373, -0.0013483, -0.0003651, 0.0003680
5: 0.0053893, 0.0059467, 0.0053893, 0.0059467, -0.0003482, 0.0003455
6: -0.0009170, 0.0012943, -0.0009170, 0.0012943, -0.0013816, 0.0013707
7: -0.0043195, -0.0013078, -0.0043195, -0.0013078, -0.0018668, 0.0018816
8: 0.9861712, 0.9882926, 0.9861712, 0.9882926, -0.0013150, 0.0013255
9: -0.0052601, -0.0033343, -0.0052601, -0.0033343, -0.0012032, 0.0011937

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005666, upper bound: 0.0005810
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005621, upper bound: 0.0005852
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0036913, 0.0046427, 0.0036913, 0.0046427, -0.0005866, 0.0005979
1: 0.0018556, 0.0019930, 0.0018556, 0.0019930, -0.0000847, 0.0000864
2: 0.0117930, 0.0123190, 0.0117930, 0.0123190, -0.0003305, 0.0003243
3: -0.0024836, -0.0019395, -0.0024836, -0.0019395, -0.0003419, 0.0003354
4: -0.0019373, -0.0013483, -0.0019373, -0.0013483, -0.0003631, 0.0003701
5: 0.0053893, 0.0059467, 0.0053893, 0.0059467, -0.0003502, 0.0003436
6: -0.0009170, 0.0012943, -0.0009170, 0.0012943, -0.0013896, 0.0013634
7: -0.0043195, -0.0013078, -0.0043195, -0.0013078, -0.0018568, 0.0018925
8: 0.9861712, 0.9882926, 0.9861712, 0.9882926, -0.0013080, 0.0013331
9: -0.0052601, -0.0033343, -0.0052601, -0.0033343, -0.0012101, 0.0011873

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005657, upper bound: 0.0005810
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005621, upper bound: 0.0005854
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0036913, 0.0046427, 0.0036913, 0.0046427, -0.0005954, 0.0005871
1: 0.0018556, 0.0019930, 0.0018556, 0.0019930, -0.0000860, 0.0000848
2: 0.0117930, 0.0123190, 0.0117930, 0.0123190, -0.0003246, 0.0003292
3: -0.0024836, -0.0019395, -0.0024836, -0.0019395, -0.0003357, 0.0003405
4: -0.0019373, -0.0013483, -0.0019373, -0.0013483, -0.0003686, 0.0003634
5: 0.0053893, 0.0059467, 0.0053893, 0.0059467, -0.0003439, 0.0003488
6: -0.0009170, 0.0012943, -0.0009170, 0.0012943, -0.0013646, 0.0013839
7: -0.0043195, -0.0013078, -0.0043195, -0.0013078, -0.0018848, 0.0018584
8: 0.9861712, 0.9882926, 0.9861712, 0.9882926, -0.0013277, 0.0013091
9: -0.0052601, -0.0033343, -0.0052601, -0.0033343, -0.0011883, 0.0012052

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005748, upper bound: 0.0005679
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005705, upper bound: 0.0005737
time: 0.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0036913, 0.0046427, 0.0036913, 0.0046427, -0.0005926, 0.0005904
1: 0.0018556, 0.0019930, 0.0018556, 0.0019930, -0.0000856, 0.0000853
2: 0.0117930, 0.0123190, 0.0117930, 0.0123190, -0.0003264, 0.0003276
3: -0.0024836, -0.0019395, -0.0024836, -0.0019395, -0.0003376, 0.0003388
4: -0.0019373, -0.0013483, -0.0019373, -0.0013483, -0.0003668, 0.0003655
5: 0.0053893, 0.0059467, 0.0053893, 0.0059467, -0.0003459, 0.0003471
6: -0.0009170, 0.0012943, -0.0009170, 0.0012943, -0.0013722, 0.0013773
7: -0.0043195, -0.0013078, -0.0043195, -0.0013078, -0.0018757, 0.0018689
8: 0.9861712, 0.9882926, 0.9861712, 0.9882926, -0.0013213, 0.0013165
9: -0.0052601, -0.0033343, -0.0052601, -0.0033343, -0.0011950, 0.0011994

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005738, upper bound: 0.0005679
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005705, upper bound: 0.0005737
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0036913, 0.0046427, 0.0036913, 0.0046427, -0.0005904, 0.0005926
1: 0.0018556, 0.0019930, 0.0018556, 0.0019930, -0.0000853, 0.0000856
2: 0.0117930, 0.0123190, 0.0117930, 0.0123190, -0.0003276, 0.0003264
3: -0.0024836, -0.0019395, -0.0024836, -0.0019395, -0.0003388, 0.0003376
4: -0.0019373, -0.0013483, -0.0019373, -0.0013483, -0.0003655, 0.0003668
5: 0.0053893, 0.0059467, 0.0053893, 0.0059467, -0.0003471, 0.0003459
6: -0.0009170, 0.0012943, -0.0009170, 0.0012943, -0.0013773, 0.0013722
7: -0.0043195, -0.0013078, -0.0043195, -0.0013078, -0.0018689, 0.0018757
8: 0.9861712, 0.9882926, 0.9861712, 0.9882926, -0.0013165, 0.0013213
9: -0.0052601, -0.0033343, -0.0052601, -0.0033343, -0.0011994, 0.0011950

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005737, upper bound: 0.0005705
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005679, upper bound: 0.0005738
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0036913, 0.0046427, 0.0036913, 0.0046427, -0.0005871, 0.0005954
1: 0.0018556, 0.0019930, 0.0018556, 0.0019930, -0.0000848, 0.0000860
2: 0.0117930, 0.0123190, 0.0117930, 0.0123190, -0.0003292, 0.0003246
3: -0.0024836, -0.0019395, -0.0024836, -0.0019395, -0.0003405, 0.0003357
4: -0.0019373, -0.0013483, -0.0019373, -0.0013483, -0.0003634, 0.0003686
5: 0.0053893, 0.0059467, 0.0053893, 0.0059467, -0.0003488, 0.0003439
6: -0.0009170, 0.0012943, -0.0009170, 0.0012943, -0.0013839, 0.0013646
7: -0.0043195, -0.0013078, -0.0043195, -0.0013078, -0.0018584, 0.0018848
8: 0.9861712, 0.9882926, 0.9861712, 0.9882926, -0.0013091, 0.0013277
9: -0.0052601, -0.0033343, -0.0052601, -0.0033343, -0.0012052, 0.0011883

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005737, upper bound: 0.0005705
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005679, upper bound: 0.0005748
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0036913, 0.0046427, 0.0036913, 0.0046427, -0.0005979, 0.0005866
1: 0.0018556, 0.0019930, 0.0018556, 0.0019930, -0.0000864, 0.0000847
2: 0.0117930, 0.0123190, 0.0117930, 0.0123190, -0.0003243, 0.0003305
3: -0.0024836, -0.0019395, -0.0024836, -0.0019395, -0.0003354, 0.0003419
4: -0.0019373, -0.0013483, -0.0019373, -0.0013483, -0.0003701, 0.0003631
5: 0.0053893, 0.0059467, 0.0053893, 0.0059467, -0.0003436, 0.0003502
6: -0.0009170, 0.0012943, -0.0009170, 0.0012943, -0.0013634, 0.0013896
7: -0.0043195, -0.0013078, -0.0043195, -0.0013078, -0.0018925, 0.0018568
8: 0.9861712, 0.9882926, 0.9861712, 0.9882926, -0.0013331, 0.0013080
9: -0.0052601, -0.0033343, -0.0052601, -0.0033343, -0.0011873, 0.0012101

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005854, upper bound: 0.0005621
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005810, upper bound: 0.0005657
time: 0.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0036913, 0.0046427, 0.0036913, 0.0046427, -0.0005944, 0.0005897
1: 0.0018556, 0.0019930, 0.0018556, 0.0019930, -0.0000859, 0.0000852
2: 0.0117930, 0.0123190, 0.0117930, 0.0123190, -0.0003260, 0.0003286
3: -0.0024836, -0.0019395, -0.0024836, -0.0019395, -0.0003372, 0.0003399
4: -0.0019373, -0.0013483, -0.0019373, -0.0013483, -0.0003680, 0.0003651
5: 0.0053893, 0.0059467, 0.0053893, 0.0059467, -0.0003455, 0.0003482
6: -0.0009170, 0.0012943, -0.0009170, 0.0012943, -0.0013707, 0.0013816
7: -0.0043195, -0.0013078, -0.0043195, -0.0013078, -0.0018816, 0.0018668
8: 0.9861712, 0.9882926, 0.9861712, 0.9882926, -0.0013255, 0.0013150
9: -0.0052601, -0.0033343, -0.0052601, -0.0033343, -0.0011937, 0.0012032

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005852, upper bound: 0.0005621
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005810, upper bound: 0.0005666
time: 0.47 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.57 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.57
Output dim: 8, lower bound: -0.0005666, upper bound: 0.0005810
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.57
Output dim: 8, lower bound: -0.0005621, upper bound: 0.0005852
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.57
Output dim: 8, lower bound: -0.0005657, upper bound: 0.0005810
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.57
Output dim: 8, lower bound: -0.0005621, upper bound: 0.0005854
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.57
Output dim: 8, lower bound: -0.0005748, upper bound: 0.0005679
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.57
Output dim: 8, lower bound: -0.0005705, upper bound: 0.0005737
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.57
Output dim: 8, lower bound: -0.0005738, upper bound: 0.0005679
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.57
Output dim: 8, lower bound: -0.0005705, upper bound: 0.0005737
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.57
Output dim: 8, lower bound: -0.0005737, upper bound: 0.0005705
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.57
Output dim: 8, lower bound: -0.0005679, upper bound: 0.0005738
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.57
Output dim: 8, lower bound: -0.0005737, upper bound: 0.0005705
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.57
Output dim: 8, lower bound: -0.0005679, upper bound: 0.0005748
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.57
Output dim: 8, lower bound: -0.0005854, upper bound: 0.0005621
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.57
Output dim: 8, lower bound: -0.0005810, upper bound: 0.0005657
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.57
Output dim: 8, lower bound: -0.0005852, upper bound: 0.0005621
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.57
Output dim: 8, lower bound: -0.0005810, upper bound: 0.0005666

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.90 + 37.94 = 40.84 seconds
