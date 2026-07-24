## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 2.776e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0001490, 0.0005931, 0.0001490, 0.0005931, -0.0002709, 0.0002709)
1: (-0.0034510, -0.0033817, -0.0034510, -0.0033817, -0.0000243, 0.0000243)
2: (0.0151291, 0.0156726, 0.0151291, 0.0156726, -0.0003157, 0.0003157)
3: (1.0067754, 1.0069143, 1.0067754, 1.0069143, -0.0001152, 0.0001152)
4: (-0.0041985, -0.0041181, -0.0041985, -0.0041181, -0.0000437, 0.0000437)
5: (0.0040941, 0.0044318, 0.0040941, 0.0044318, -0.0002048, 0.0002048)
6: (-0.0025994, -0.0025720, -0.0025994, -0.0025720, -0.0000230, 0.0000230)
7: (-0.0125696, -0.0116633, -0.0125696, -0.0116633, -0.0006433, 0.0006433)
8: (-0.0131573, -0.0123408, -0.0131573, -0.0123408, -0.0004219, 0.0004219)
9: (0.0019942, 0.0023714, 0.0019942, 0.0023714, -0.0001790, 0.0001790)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.53 + 1.27 = 2.80 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0000340, upper bound: 0.0000340

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000338, upper bound: 0.0000325
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000325, upper bound: 0.0000338
time: 0.42 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.88 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.88
Output dim: 3, lower bound: -0.0000338, upper bound: 0.0000325
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.88
Output dim: 3, lower bound: -0.0000325, upper bound: 0.0000338

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0001490, 0.0005931, 0.0001490, 0.0005931, -0.0002674, 0.0002679
1: -0.0034510, -0.0033817, -0.0034510, -0.0033817, -0.0000233, 0.0000235
2: 0.0151291, 0.0156726, 0.0151291, 0.0156726, -0.0003111, 0.0003118
3: 1.0067754, 1.0069143, 1.0067754, 1.0069143, -0.0001147, 0.0001145
4: -0.0041985, -0.0041181, -0.0041985, -0.0041181, -0.0000431, 0.0000430
5: 0.0040941, 0.0044318, 0.0040941, 0.0044318, -0.0002020, 0.0002025
6: -0.0025994, -0.0025720, -0.0025994, -0.0025720, -0.0000229, 0.0000229
7: -0.0125696, -0.0116633, -0.0125696, -0.0116633, -0.0006387, 0.0006377
8: -0.0131573, -0.0123408, -0.0131573, -0.0123408, -0.0004150, 0.0004137
9: 0.0019942, 0.0023714, 0.0019942, 0.0023714, -0.0001750, 0.0001757

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 253

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000338, upper bound: 0.0000320
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000337, upper bound: 0.0000325
time: 0.45 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0001490, 0.0005931, 0.0001490, 0.0005931, -0.0002679, 0.0002674
1: -0.0034510, -0.0033817, -0.0034510, -0.0033817, -0.0000235, 0.0000233
2: 0.0151291, 0.0156726, 0.0151291, 0.0156726, -0.0003118, 0.0003111
3: 1.0067754, 1.0069143, 1.0067754, 1.0069143, -0.0001145, 0.0001147
4: -0.0041985, -0.0041181, -0.0041985, -0.0041181, -0.0000430, 0.0000431
5: 0.0040941, 0.0044318, 0.0040941, 0.0044318, -0.0002025, 0.0002020
6: -0.0025994, -0.0025720, -0.0025994, -0.0025720, -0.0000229, 0.0000229
7: -0.0125696, -0.0116633, -0.0125696, -0.0116633, -0.0006377, 0.0006387
8: -0.0131573, -0.0123408, -0.0131573, -0.0123408, -0.0004137, 0.0004150
9: 0.0019942, 0.0023714, 0.0019942, 0.0023714, -0.0001757, 0.0001750

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000299, upper bound: 0.0000275
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000260, upper bound: 0.0000320
time: 0.47 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.03 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 3, lower bound: -0.0000338, upper bound: 0.0000320
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 3, lower bound: -0.0000337, upper bound: 0.0000325
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 3, lower bound: -0.0000299, upper bound: 0.0000275
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 3, lower bound: -0.0000260, upper bound: 0.0000320

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0001490, 0.0005931, 0.0001490, 0.0005931, -0.0002649, 0.0002664
1: -0.0034510, -0.0033817, -0.0034510, -0.0033817, -0.0000231, 0.0000234
2: 0.0151291, 0.0156726, 0.0151291, 0.0156726, -0.0003086, 0.0003105
3: 1.0067754, 1.0069143, 1.0067754, 1.0069143, -0.0001113, 0.0001102
4: -0.0041985, -0.0041181, -0.0041985, -0.0041181, -0.0000429, 0.0000427
5: 0.0040941, 0.0044318, 0.0040941, 0.0044318, -0.0002002, 0.0002013
6: -0.0025994, -0.0025720, -0.0025994, -0.0025720, -0.0000224, 0.0000224
7: -0.0125696, -0.0116633, -0.0125696, -0.0116633, -0.0006327, 0.0006296
8: -0.0131573, -0.0123408, -0.0131573, -0.0123408, -0.0004138, 0.0004115
9: 0.0019942, 0.0023714, 0.0019942, 0.0023714, -0.0001745, 0.0001754

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000338, upper bound: 0.0000319
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000330, upper bound: 0.0000320
time: 0.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0001490, 0.0005931, 0.0001490, 0.0005931, -0.0002658, 0.0002655
1: -0.0034510, -0.0033817, -0.0034510, -0.0033817, -0.0000232, 0.0000233
2: 0.0151291, 0.0156726, 0.0151291, 0.0156726, -0.0003097, 0.0003094
3: 1.0067754, 1.0069143, 1.0067754, 1.0069143, -0.0001104, 0.0001111
4: -0.0041985, -0.0041181, -0.0041985, -0.0041181, -0.0000428, 0.0000428
5: 0.0040941, 0.0044318, 0.0040941, 0.0044318, -0.0002008, 0.0002006
6: -0.0025994, -0.0025720, -0.0025994, -0.0025720, -0.0000224, 0.0000224
7: -0.0125696, -0.0116633, -0.0125696, -0.0116633, -0.0006306, 0.0006316
8: -0.0131573, -0.0123408, -0.0131573, -0.0123408, -0.0004128, 0.0004125
9: 0.0019942, 0.0023714, 0.0019942, 0.0023714, -0.0001747, 0.0001751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000294, upper bound: 0.0000252
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000265, upper bound: 0.0000280
time: 0.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0001490, 0.0005931, 0.0001490, 0.0005931, -0.0002515, 0.0002505
1: -0.0034510, -0.0033817, -0.0034510, -0.0033817, -0.0000204, 0.0000206
2: 0.0151291, 0.0156726, 0.0151291, 0.0156726, -0.0002969, 0.0002956
3: 1.0067754, 1.0069143, 1.0067754, 1.0069143, -0.0000903, 0.0000883
4: -0.0041985, -0.0041181, -0.0041985, -0.0041181, -0.0000415, 0.0000417
5: 0.0040941, 0.0044318, 0.0040941, 0.0044318, -0.0001904, 0.0001896
6: -0.0025994, -0.0025720, -0.0025994, -0.0025720, -0.0000196, 0.0000192
7: -0.0125696, -0.0116633, -0.0125696, -0.0116633, -0.0005717, 0.0005736
8: -0.0131573, -0.0123408, -0.0131573, -0.0123408, -0.0004048, 0.0004060
9: 0.0019942, 0.0023714, 0.0019942, 0.0023714, -0.0001750, 0.0001743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000299, upper bound: 0.0000253
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000296, upper bound: 0.0000274
time: 0.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0001490, 0.0005931, 0.0001490, 0.0005931, -0.0002511, 0.0002508
1: -0.0034510, -0.0033817, -0.0034510, -0.0033817, -0.0000208, 0.0000202
2: 0.0151291, 0.0156726, 0.0151291, 0.0156726, -0.0002964, 0.0002961
3: 1.0067754, 1.0069143, 1.0067754, 1.0069143, -0.0000881, 0.0000906
4: -0.0041985, -0.0041181, -0.0041985, -0.0041181, -0.0000416, 0.0000417
5: 0.0040941, 0.0044318, 0.0040941, 0.0044318, -0.0001901, 0.0001899
6: -0.0025994, -0.0025720, -0.0025994, -0.0025720, -0.0000193, 0.0000196
7: -0.0125696, -0.0116633, -0.0125696, -0.0116633, -0.0005725, 0.0005727
8: -0.0131573, -0.0123408, -0.0131573, -0.0123408, -0.0004046, 0.0004061
9: 0.0019942, 0.0023714, 0.0019942, 0.0023714, -0.0001750, 0.0001743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 253

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000260, upper bound: 0.0000320
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000244, upper bound: 0.0000319
time: 0.47 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 1.97 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 1.97
Output dim: 3, lower bound: -0.0000338, upper bound: 0.0000319
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 1.97
Output dim: 3, lower bound: -0.0000330, upper bound: 0.0000320
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 1.97
Output dim: 3, lower bound: -0.0000294, upper bound: 0.0000252
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 1.97
Output dim: 3, lower bound: -0.0000265, upper bound: 0.0000280
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 1.97
Output dim: 3, lower bound: -0.0000299, upper bound: 0.0000253
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 1.97
Output dim: 3, lower bound: -0.0000296, upper bound: 0.0000274
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 1.97
Output dim: 3, lower bound: -0.0000260, upper bound: 0.0000320
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 1.97
Output dim: 3, lower bound: -0.0000244, upper bound: 0.0000319

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0001490, 0.0005931, 0.0001490, 0.0005931, -0.0002600, 0.0002616
1: -0.0034510, -0.0033817, -0.0034510, -0.0033817, -0.0000216, 0.0000220
2: 0.0151291, 0.0156726, 0.0151291, 0.0156726, -0.0003022, 0.0003043
3: 1.0067754, 1.0069143, 1.0067754, 1.0069143, -0.0001115, 0.0001101
4: -0.0041985, -0.0041181, -0.0041985, -0.0041181, -0.0000418, 0.0000415
5: 0.0040941, 0.0044318, 0.0040941, 0.0044318, -0.0001964, 0.0001976
6: -0.0025994, -0.0025720, -0.0025994, -0.0025720, -0.0000224, 0.0000223
7: -0.0125696, -0.0116633, -0.0125696, -0.0116633, -0.0006245, 0.0006220
8: -0.0131573, -0.0123408, -0.0131573, -0.0123408, -0.0004008, 0.0003975
9: 0.0019942, 0.0023714, 0.0019942, 0.0023714, -0.0001669, 0.0001683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000318, upper bound: 0.0000243
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000274, upper bound: 0.0000296
time: 0.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0001490, 0.0005931, 0.0001490, 0.0005931, -0.0002596, 0.0002615
1: -0.0034510, -0.0033817, -0.0034510, -0.0033817, -0.0000217, 0.0000218
2: 0.0151291, 0.0156726, 0.0151291, 0.0156726, -0.0003019, 0.0003040
3: 1.0067754, 1.0069143, 1.0067754, 1.0069143, -0.0001111, 0.0001103
4: -0.0041985, -0.0041181, -0.0041985, -0.0041181, -0.0000418, 0.0000415
5: 0.0040941, 0.0044318, 0.0040941, 0.0044318, -0.0001961, 0.0001976
6: -0.0025994, -0.0025720, -0.0025994, -0.0025720, -0.0000223, 0.0000224
7: -0.0125696, -0.0116633, -0.0125696, -0.0116633, -0.0006251, 0.0006213
8: -0.0131573, -0.0123408, -0.0131573, -0.0123408, -0.0003998, 0.0003975
9: 0.0019942, 0.0023714, 0.0019942, 0.0023714, -0.0001669, 0.0001678

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000310, upper bound: 0.0000244
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000252, upper bound: 0.0000296
time: 0.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0001490, 0.0005931, 0.0001490, 0.0005931, -0.0001677, 0.0001479
1: -0.0034510, -0.0033817, -0.0034510, -0.0033817, -0.0000236, 0.0000235
2: 0.0151291, 0.0156726, 0.0151291, 0.0156726, -0.0001996, 0.0001771
3: 1.0067754, 1.0069143, 1.0067754, 1.0069143, -0.0000689, 0.0000673
4: -0.0041985, -0.0041181, -0.0041985, -0.0041181, -0.0000256, 0.0000285
5: 0.0040941, 0.0044318, 0.0040941, 0.0044318, -0.0001271, 0.0001122
6: -0.0025994, -0.0025720, -0.0025994, -0.0025720, -0.0000135, 0.0000114
7: -0.0125696, -0.0116633, -0.0125696, -0.0116633, -0.0003289, 0.0003842
8: -0.0131573, -0.0123408, -0.0131573, -0.0123408, -0.0002569, 0.0002818
9: 0.0019942, 0.0023714, 0.0019942, 0.0023714, -0.0001257, 0.0001167

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.ADV_EXAMPLE
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000286, upper bound: 0.0000252
time: 0.44 seconds

## RS Result
status: Status.ADV_EXAMPLE
execution time: (base) + (rs) = 2.80 + 18.62 = 21.41 seconds
