## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0051876


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004659, 0.0004659)
1: (0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0025796, 0.0025796)
2: (0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0057630, 0.0057630)
3: (0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0024285, 0.0024285)
4: (1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0094218, 0.0094218)
5: (0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0018329, 0.0018329)
6: (-0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0023853, 0.0023853)
7: (-0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0003043, 0.0003043)
8: (-0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0016480, 0.0016480)
9: (-0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0082505, 0.0082504)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.55 + 1.65 = 3.19 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0066253, upper bound: 0.0066253

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0065199, upper bound: 0.0065348
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0065347, upper bound: 0.0065199
time: 1.08 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.01 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.01
Output dim: 4, lower bound: -0.0065199, upper bound: 0.0065348
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.01
Output dim: 4, lower bound: -0.0065347, upper bound: 0.0065199

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004524, 0.0004529
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0025078, 0.0025049
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0055963, 0.0056028
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0023610, 0.0023583
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0091599, 0.0091492
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017820, 0.0017799
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0023163, 0.0023190
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002955, 0.0002958
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0016022, 0.0016003
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0080117, 0.0080211

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0059812, upper bound: 0.0058731
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0058347, upper bound: 0.0059983
time: 0.81 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004529, 0.0004524
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0025049, 0.0025078
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0056028, 0.0055963
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0023583, 0.0023610
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0091492, 0.0091599
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017799, 0.0017820
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0023190, 0.0023163
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002958, 0.0002955
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0016003, 0.0016022
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0080211, 0.0080117

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0059983, upper bound: 0.0058348
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0058731, upper bound: 0.0059812
time: 0.87 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.25 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.25
Output dim: 4, lower bound: -0.0059812, upper bound: 0.0058731
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.25
Output dim: 4, lower bound: -0.0058347, upper bound: 0.0059983
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.25
Output dim: 4, lower bound: -0.0059983, upper bound: 0.0058348
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.25
Output dim: 4, lower bound: -0.0058731, upper bound: 0.0059812

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004391, 0.0004598
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0025458, 0.0024315
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0054323, 0.0056877
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0023968, 0.0022892
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0092987, 0.0088812
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0018090, 0.0017277
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022484, 0.0023541
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002868, 0.0003003
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0016265, 0.0015535
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0077770, 0.0081427

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0059786, upper bound: 0.0057688
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0058907, upper bound: 0.0058704
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004524, 0.0004397
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0024345, 0.0025049
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0055963, 0.0054388
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022919, 0.0023583
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0088919, 0.0091492
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017298, 0.0017799
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0023163, 0.0022511
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002955, 0.0002871
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015553, 0.0016003
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0080117, 0.0077864

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0058321, upper bound: 0.0058938
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057573, upper bound: 0.0059957
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004397, 0.0004574
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0025324, 0.0024345
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0054388, 0.0056576
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0023841, 0.0022919
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0092495, 0.0088919
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017994, 0.0017298
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022511, 0.0023416
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002871, 0.0002987
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0016179, 0.0015553
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0077864, 0.0080995

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0059957, upper bound: 0.0057573
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0058938, upper bound: 0.0058321
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004529, 0.0004391
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0024315, 0.0025078
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0056028, 0.0054323
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022892, 0.0023610
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0088812, 0.0091599
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017277, 0.0017820
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0023190, 0.0022484
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002958, 0.0002868
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015535, 0.0016022
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0080211, 0.0077770

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0058704, upper bound: 0.0058907
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057688, upper bound: 0.0059786
time: 0.80 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.11 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 4, lower bound: -0.0059786, upper bound: 0.0057688
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 4, lower bound: -0.0058907, upper bound: 0.0058704
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 4, lower bound: -0.0058321, upper bound: 0.0058938
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 4, lower bound: -0.0057573, upper bound: 0.0059957
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 4, lower bound: -0.0059957, upper bound: 0.0057573
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 4, lower bound: -0.0058938, upper bound: 0.0058321
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 4, lower bound: -0.0058704, upper bound: 0.0058907
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 4, lower bound: -0.0057688, upper bound: 0.0059786

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004390, 0.0004601
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0025478, 0.0024306
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0054303, 0.0056921
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0023987, 0.0022883
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0093059, 0.0088778
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0018104, 0.0017271
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022476, 0.0023559
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002867, 0.0003005
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0016278, 0.0015529
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0077741, 0.0081490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0059749, upper bound: 0.0057651
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0059749, upper bound: 0.0057644
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004397, 0.0004596
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0025449, 0.0024346
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0054393, 0.0056857
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0023959, 0.0022921
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0092954, 0.0088925
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0018083, 0.0017300
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022513, 0.0023533
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002872, 0.0003002
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0016259, 0.0015555
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0077870, 0.0081397

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0058870, upper bound: 0.0058665
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0058857, upper bound: 0.0058604
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004524, 0.0004402
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0024375, 0.0025048
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0055960, 0.0054457
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022948, 0.0023582
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0089030, 0.0091489
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017320, 0.0017798
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0023162, 0.0022539
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002954, 0.0002875
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015573, 0.0016003
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0080114, 0.0077962

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0058230, upper bound: 0.0058884
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0058282, upper bound: 0.0058902
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004531, 0.0004395
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0024335, 0.0025089
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0056050, 0.0054368
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022911, 0.0023620
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0088885, 0.0091636
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017292, 0.0017827
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0023199, 0.0022503
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002959, 0.0002870
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015547, 0.0016029
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0080243, 0.0077834

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057514, upper bound: 0.0059918
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057535, upper bound: 0.0059921
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004395, 0.0004578
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0025348, 0.0024335
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0054368, 0.0056631
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0023865, 0.0022911
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0092585, 0.0088885
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0018012, 0.0017292
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022503, 0.0023439
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002870, 0.0002990
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0016195, 0.0015547
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0077834, 0.0081075

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0059921, upper bound: 0.0057535
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0059918, upper bound: 0.0057514
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004402, 0.0004572
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0025314, 0.0024375
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0054457, 0.0056555
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0023833, 0.0022948
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0092461, 0.0089030
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017987, 0.0017320
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022539, 0.0023408
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002875, 0.0002986
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0016173, 0.0015573
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0077962, 0.0080966

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0058902, upper bound: 0.0058282
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0058884, upper bound: 0.0058230
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004529, 0.0004397
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0024346, 0.0025077
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0056026, 0.0054393
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022921, 0.0023609
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0088925, 0.0091595
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017300, 0.0017819
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0023189, 0.0022513
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002958, 0.0002872
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015555, 0.0016022
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0080208, 0.0077870

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0058604, upper bound: 0.0058857
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0058665, upper bound: 0.0058870
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004536, 0.0004390
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0024306, 0.0025117
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0056115, 0.0054303
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022883, 0.0023647
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0088778, 0.0091741
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017271, 0.0017847
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0023226, 0.0022476
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002963, 0.0002867
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015529, 0.0016047
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0080335, 0.0077741

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057644, upper bound: 0.0059749
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057651, upper bound: 0.0059749
time: 0.82 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.15 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 4, lower bound: -0.0059749, upper bound: 0.0057651
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 4, lower bound: -0.0059749, upper bound: 0.0057644
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 4, lower bound: -0.0058870, upper bound: 0.0058665
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 4, lower bound: -0.0058857, upper bound: 0.0058604
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 4, lower bound: -0.0058230, upper bound: 0.0058884
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 4, lower bound: -0.0058282, upper bound: 0.0058902
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 4, lower bound: -0.0057514, upper bound: 0.0059918
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 4, lower bound: -0.0057535, upper bound: 0.0059921
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 4, lower bound: -0.0059921, upper bound: 0.0057535
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 4, lower bound: -0.0059918, upper bound: 0.0057514
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 4, lower bound: -0.0058902, upper bound: 0.0058282
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 4, lower bound: -0.0058884, upper bound: 0.0058230
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 4, lower bound: -0.0058604, upper bound: 0.0058857
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 4, lower bound: -0.0058665, upper bound: 0.0058870
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 4, lower bound: -0.0057644, upper bound: 0.0059749
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 4, lower bound: -0.0057651, upper bound: 0.0059749

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004388, 0.0004600
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0025472, 0.0024299
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0054286, 0.0056908
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0023981, 0.0022876
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0093038, 0.0088752
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0018100, 0.0017266
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022469, 0.0023554
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002866, 0.0003005
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0016274, 0.0015524
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0077718, 0.0081471

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0059338, upper bound: 0.0055535
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057116, upper bound: 0.0057228
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004388, 0.0004600
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0025471, 0.0024298
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0054284, 0.0056905
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0023980, 0.0022875
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0093033, 0.0088748
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0018099, 0.0017265
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022468, 0.0023553
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002866, 0.0003004
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0016273, 0.0015523
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0077714, 0.0081466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0059338, upper bound: 0.0055521
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057116, upper bound: 0.0057220
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004396, 0.0004595
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0025444, 0.0024339
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0054376, 0.0056844
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0023954, 0.0022914
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0092933, 0.0088899
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0018079, 0.0017294
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022506, 0.0023527
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002871, 0.0003001
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0016255, 0.0015550
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0077847, 0.0081379

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0058456, upper bound: 0.0056365
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0056407, upper bound: 0.0058247
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004396, 0.0004595
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0025442, 0.0024338
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0054374, 0.0056840
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0023953, 0.0022913
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0092927, 0.0088895
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0018078, 0.0017294
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022505, 0.0023526
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002871, 0.0003001
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0016254, 0.0015549
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0077843, 0.0081374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0058445, upper bound: 0.0056335
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0056382, upper bound: 0.0058180
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004522, 0.0004401
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0024366, 0.0025040
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0055942, 0.0054437
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022940, 0.0023574
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0088999, 0.0091458
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017314, 0.0017792
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0023154, 0.0022531
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002954, 0.0002874
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015567, 0.0015998
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0080088, 0.0077934

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057808, upper bound: 0.0056497
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055979, upper bound: 0.0058473
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004522, 0.0004401
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0024368, 0.0025039
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0055939, 0.0054441
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022941, 0.0023573
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0089004, 0.0091454
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017315, 0.0017792
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0023153, 0.0022533
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002953, 0.0002874
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015568, 0.0015997
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0080084, 0.0077939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057857, upper bound: 0.0056503
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0056098, upper bound: 0.0058490
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004530, 0.0004393
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0024327, 0.0025080
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0056032, 0.0054348
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022903, 0.0023612
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0088853, 0.0091606
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017285, 0.0017821
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0023191, 0.0022494
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002958, 0.0002869
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015542, 0.0016023
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0080217, 0.0077806

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057097, upper bound: 0.0057341
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055356, upper bound: 0.0059507
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004529, 0.0004394
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0024328, 0.0025079
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0056030, 0.0054352
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022904, 0.0023611
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0088858, 0.0091602
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017287, 0.0017820
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0023190, 0.0022496
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002958, 0.0002870
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015543, 0.0016023
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0080213, 0.0077811

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057119, upper bound: 0.0057340
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055369, upper bound: 0.0059508
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004394, 0.0004577
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0025343, 0.0024328
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0054352, 0.0056618
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0023859, 0.0022904
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0092564, 0.0088858
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0018007, 0.0017287
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022496, 0.0023434
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002870, 0.0002989
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0016191, 0.0015543
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0077811, 0.0081056

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0059508, upper bound: 0.0055369
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057340, upper bound: 0.0057119
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004393, 0.0004577
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0025341, 0.0024327
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0054348, 0.0056615
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0023858, 0.0022903
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0092559, 0.0088853
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0018006, 0.0017285
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022494, 0.0023433
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002869, 0.0002989
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0016190, 0.0015542
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0077806, 0.0081051

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0059507, upper bound: 0.0055356
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057341, upper bound: 0.0057097
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004401, 0.0004571
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0025309, 0.0024368
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0054441, 0.0056543
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0023827, 0.0022941
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0092441, 0.0089004
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017983, 0.0017315
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022533, 0.0023403
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002874, 0.0002985
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0016169, 0.0015568
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0077939, 0.0080948

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0058490, upper bound: 0.0056098
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0056503, upper bound: 0.0057857
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004401, 0.0004571
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0025307, 0.0024366
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0054437, 0.0056539
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0023826, 0.0022940
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0092435, 0.0088999
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017982, 0.0017314
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022531, 0.0023401
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002874, 0.0002985
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0016168, 0.0015567
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0077934, 0.0080943

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0058473, upper bound: 0.0055979
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0056497, upper bound: 0.0057808
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004528, 0.0004396
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0024338, 0.0025069
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0056007, 0.0054374
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022913, 0.0023602
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0088895, 0.0091565
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017294, 0.0017813
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0023181, 0.0022505
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002957, 0.0002871
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015549, 0.0016016
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0080181, 0.0077843

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0058180, upper bound: 0.0056382
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0056335, upper bound: 0.0058445
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004527, 0.0004396
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0024339, 0.0025068
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0056004, 0.0054376
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022914, 0.0023600
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0088899, 0.0091560
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017294, 0.0017812
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0023180, 0.0022506
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002957, 0.0002871
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015550, 0.0016015
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0080177, 0.0077847

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0058247, upper bound: 0.0056407
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0056365, upper bound: 0.0058456
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004535, 0.0004388
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0024298, 0.0025109
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0056096, 0.0054284
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022875, 0.0023639
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0088748, 0.0091711
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017265, 0.0017841
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0023218, 0.0022468
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002962, 0.0002866
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015523, 0.0016042
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0080309, 0.0077714

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057220, upper bound: 0.0057116
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055521, upper bound: 0.0059338
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004535, 0.0004388
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0024299, 0.0025108
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0056093, 0.0054286
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022876, 0.0023638
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0088752, 0.0091705
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017266, 0.0017840
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0023217, 0.0022469
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002961, 0.0002866
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015524, 0.0016041
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0080304, 0.0077718

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057228, upper bound: 0.0057116
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055535, upper bound: 0.0059339
time: 0.92 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.45 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 4, lower bound: -0.0059338, upper bound: 0.0055535
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 4, lower bound: -0.0057116, upper bound: 0.0057228
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 4, lower bound: -0.0059338, upper bound: 0.0055521
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 4, lower bound: -0.0057116, upper bound: 0.0057220
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 4, lower bound: -0.0058456, upper bound: 0.0056365
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 4, lower bound: -0.0056407, upper bound: 0.0058247
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 4, lower bound: -0.0058445, upper bound: 0.0056335
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 4, lower bound: -0.0056382, upper bound: 0.0058180
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 4, lower bound: -0.0057808, upper bound: 0.0056497
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 4, lower bound: -0.0055979, upper bound: 0.0058473
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 4, lower bound: -0.0057857, upper bound: 0.0056503
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 4, lower bound: -0.0056098, upper bound: 0.0058490
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 4, lower bound: -0.0057097, upper bound: 0.0057341
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 4, lower bound: -0.0055356, upper bound: 0.0059507
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 4, lower bound: -0.0057119, upper bound: 0.0057340
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 4, lower bound: -0.0055369, upper bound: 0.0059508
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 4, lower bound: -0.0059508, upper bound: 0.0055369
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 4, lower bound: -0.0057340, upper bound: 0.0057119
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 4, lower bound: -0.0059507, upper bound: 0.0055356
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 4, lower bound: -0.0057341, upper bound: 0.0057097
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 4, lower bound: -0.0058490, upper bound: 0.0056098
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 4, lower bound: -0.0056503, upper bound: 0.0057857
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 4, lower bound: -0.0058473, upper bound: 0.0055979
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 4, lower bound: -0.0056497, upper bound: 0.0057808
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 4, lower bound: -0.0058180, upper bound: 0.0056382
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 4, lower bound: -0.0056335, upper bound: 0.0058445
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 4, lower bound: -0.0058247, upper bound: 0.0056407
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 4, lower bound: -0.0056365, upper bound: 0.0058456
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 4, lower bound: -0.0057220, upper bound: 0.0057116
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 4, lower bound: -0.0055521, upper bound: 0.0059338
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 4, lower bound: -0.0057228, upper bound: 0.0057116
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 4, lower bound: -0.0055535, upper bound: 0.0059339

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004349, 0.0004595
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0025441, 0.0024079
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0053795, 0.0056838
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0023952, 0.0022670
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0092923, 0.0087949
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0018077, 0.0017110
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022266, 0.0023525
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002840, 0.0003001
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0016254, 0.0015384
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0077015, 0.0081370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0056508, upper bound: 0.0049772
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053196, upper bound: 0.0052144
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004383, 0.0004561
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0025253, 0.0024266
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0054213, 0.0056417
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0023774, 0.0022845
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0092236, 0.0088631
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017944, 0.0017242
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022438, 0.0023351
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002862, 0.0002979
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0016134, 0.0015503
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0077612, 0.0080769

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053916, upper bound: 0.0051315
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051314, upper bound: 0.0054260
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004349, 0.0004595
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0025440, 0.0024078
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0053793, 0.0056835
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0023950, 0.0022668
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0092919, 0.0087945
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0018076, 0.0017109
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022265, 0.0023524
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002840, 0.0003001
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0016253, 0.0015383
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0077011, 0.0081366

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0056508, upper bound: 0.0049728
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053193, upper bound: 0.0052143
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004382, 0.0004560
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0025251, 0.0024265
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0054210, 0.0056414
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0023773, 0.0022844
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0092230, 0.0088627
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017942, 0.0017242
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022437, 0.0023349
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002862, 0.0002978
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0016133, 0.0015502
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0077609, 0.0080763

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053916, upper bound: 0.0051309
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051313, upper bound: 0.0054249
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004356, 0.0004589
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0025411, 0.0024119
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0053885, 0.0056770
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0023923, 0.0022707
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0092812, 0.0088096
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0018056, 0.0017138
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022303, 0.0023497
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002845, 0.0002997
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0016234, 0.0015409
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0077144, 0.0081273

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055451, upper bound: 0.0050259
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052691, upper bound: 0.0053318
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004390, 0.0004556
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0025224, 0.0024309
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0054309, 0.0056353
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0023747, 0.0022886
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0092130, 0.0088788
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017923, 0.0017273
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022478, 0.0023324
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002867, 0.0002975
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0016115, 0.0015531
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0077750, 0.0080676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053077, upper bound: 0.0051941
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0050916, upper bound: 0.0055576
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004356, 0.0004589
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0025409, 0.0024118
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0053883, 0.0056767
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0023922, 0.0022706
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0092808, 0.0088092
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0018055, 0.0017137
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022302, 0.0023496
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002845, 0.0002997
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0016234, 0.0015409
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0077140, 0.0081269

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055442, upper bound: 0.0050138
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052645, upper bound: 0.0053289
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004390, 0.0004555
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0025222, 0.0024308
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0054307, 0.0056349
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0023746, 0.0022885
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0092124, 0.0088785
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017922, 0.0017272
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022477, 0.0023323
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002867, 0.0002975
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0016114, 0.0015530
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0077747, 0.0080671

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053063, upper bound: 0.0051871
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0050905, upper bound: 0.0055480
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004481, 0.0004396
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0024343, 0.0024811
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0055431, 0.0054384
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022918, 0.0023359
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0088912, 0.0090623
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017297, 0.0017630
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022942, 0.0022509
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002927, 0.0002871
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015552, 0.0015851
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0079356, 0.0077858

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055127, upper bound: 0.0050973
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051443, upper bound: 0.0053134
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004515, 0.0004361
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0024147, 0.0024998
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0055848, 0.0053946
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022733, 0.0023534
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0088196, 0.0091304
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017158, 0.0017762
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0023115, 0.0022328
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002949, 0.0002848
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015427, 0.0015971
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0079953, 0.0077231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052972, upper bound: 0.0052689
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049894, upper bound: 0.0055447
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004481, 0.0004397
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0024344, 0.0024810
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0055428, 0.0054387
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022919, 0.0023358
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0088916, 0.0090618
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017298, 0.0017629
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022941, 0.0022510
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002926, 0.0002871
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015553, 0.0015851
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0079352, 0.0077862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055170, upper bound: 0.0050986
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051509, upper bound: 0.0053149
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004515, 0.0004361
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0024148, 0.0024997
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0055846, 0.0053950
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022735, 0.0023533
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0088201, 0.0091301
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017159, 0.0017762
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0023114, 0.0022329
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002948, 0.0002848
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015428, 0.0015970
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0079950, 0.0077236

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053017, upper bound: 0.0052730
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049957, upper bound: 0.0055459
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004488, 0.0004389
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0024303, 0.0024851
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0055521, 0.0054296
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022881, 0.0023397
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0088768, 0.0090770
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017269, 0.0017658
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022980, 0.0022473
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002931, 0.0002867
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015527, 0.0015877
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0079485, 0.0077732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054165, upper bound: 0.0051465
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051081, upper bound: 0.0054188
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004522, 0.0004354
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0024107, 0.0025041
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0055944, 0.0053857
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022696, 0.0023575
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0088050, 0.0091462
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017129, 0.0017793
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0023155, 0.0022291
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002954, 0.0002843
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015401, 0.0015998
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0080091, 0.0077103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052051, upper bound: 0.0053357
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049595, upper bound: 0.0056662
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004488, 0.0004390
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0024305, 0.0024850
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0055518, 0.0054299
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022882, 0.0023396
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0088773, 0.0090766
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017270, 0.0017658
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022979, 0.0022474
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002931, 0.0002867
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015528, 0.0015876
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0079481, 0.0077736

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054178, upper bound: 0.0051468
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051090, upper bound: 0.0054190
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004522, 0.0004354
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0024108, 0.0025040
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0055942, 0.0053861
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022697, 0.0023574
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0088056, 0.0091459
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017130, 0.0017792
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0023154, 0.0022293
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002954, 0.0002844
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015402, 0.0015998
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0080088, 0.0077108

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052051, upper bound: 0.0053357
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049614, upper bound: 0.0056644
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004354, 0.0004569
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0025300, 0.0024108
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0053861, 0.0056522
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0023819, 0.0022697
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0092407, 0.0088056
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017977, 0.0017130
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022293, 0.0023394
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002844, 0.0002984
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0016163, 0.0015402
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0077108, 0.0080918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0056644, upper bound: 0.0049614
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053357, upper bound: 0.0052051
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004390, 0.0004537
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0025123, 0.0024305
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0054299, 0.0056127
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0023652, 0.0022882
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0091762, 0.0088773
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017851, 0.0017270
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022474, 0.0023231
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002867, 0.0002963
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0016051, 0.0015528
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0077736, 0.0080353

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054190, upper bound: 0.0051090
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051468, upper bound: 0.0054178
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004354, 0.0004569
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0025298, 0.0024107
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0053857, 0.0056519
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0023817, 0.0022696
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0092402, 0.0088050
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017976, 0.0017129
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022291, 0.0023393
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002843, 0.0002984
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0016163, 0.0015401
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0077103, 0.0080914

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0056662, upper bound: 0.0049595
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053357, upper bound: 0.0052051
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004389, 0.0004537
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0025121, 0.0024303
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0054296, 0.0056124
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0023651, 0.0022881
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0091756, 0.0088768
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017850, 0.0017269
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022473, 0.0023229
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002867, 0.0002963
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0016050, 0.0015527
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0077732, 0.0080348

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054188, upper bound: 0.0051081
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051465, upper bound: 0.0054165
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004361, 0.0004563
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0025266, 0.0024148
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0053950, 0.0056448
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0023787, 0.0022735
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0092286, 0.0088201
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017953, 0.0017159
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022329, 0.0023364
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002848, 0.0002980
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0016142, 0.0015428
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0077236, 0.0080812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055459, upper bound: 0.0049957
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052730, upper bound: 0.0053017
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004397, 0.0004531
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0025089, 0.0024344
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0054387, 0.0056052
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0023620, 0.0022919
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0091638, 0.0088916
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017827, 0.0017298
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022510, 0.0023200
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002871, 0.0002959
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0016029, 0.0015553
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0077862, 0.0080245

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053149, upper bound: 0.0051509
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0050986, upper bound: 0.0055170
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004361, 0.0004563
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0025265, 0.0024147
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0053946, 0.0056445
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0023786, 0.0022733
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0092281, 0.0088196
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017952, 0.0017158
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022328, 0.0023362
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002848, 0.0002980
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0016141, 0.0015427
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0077231, 0.0080808

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055447, upper bound: 0.0049894
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052689, upper bound: 0.0052972
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004396, 0.0004531
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0025087, 0.0024343
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0054384, 0.0056048
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0023619, 0.0022918
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0091632, 0.0088912
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017826, 0.0017297
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022509, 0.0023198
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002871, 0.0002959
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0016028, 0.0015552
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0077858, 0.0080240

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053134, upper bound: 0.0051443
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0050973, upper bound: 0.0055127
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004486, 0.0004390
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0024308, 0.0024840
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0055496, 0.0054307
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022885, 0.0023386
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0088785, 0.0090729
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017272, 0.0017650
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022969, 0.0022477
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002930, 0.0002867
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015530, 0.0015870
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0079449, 0.0077747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055480, upper bound: 0.0050905
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051871, upper bound: 0.0053063
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004522, 0.0004356
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0024118, 0.0025037
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0055934, 0.0053883
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022706, 0.0023571
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0088092, 0.0091446
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017137, 0.0017790
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0023151, 0.0022302
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002953, 0.0002845
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015409, 0.0015995
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0080077, 0.0077140

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053289, upper bound: 0.0052645
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0050138, upper bound: 0.0055442
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004486, 0.0004390
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0024309, 0.0024839
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0055493, 0.0054309
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022886, 0.0023385
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0088788, 0.0090724
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017273, 0.0017649
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022968, 0.0022478
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002930, 0.0002867
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015531, 0.0015869
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0079445, 0.0077750

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055576, upper bound: 0.0050916
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051941, upper bound: 0.0053077
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004521, 0.0004356
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0024119, 0.0025035
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0055931, 0.0053885
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022707, 0.0023570
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0088096, 0.0091441
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017138, 0.0017789
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0023150, 0.0022303
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002953, 0.0002845
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015409, 0.0015995
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0080073, 0.0077144

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053318, upper bound: 0.0052691
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0050259, upper bound: 0.0055451
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004493, 0.0004382
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0024265, 0.0024880
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0055585, 0.0054210
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022844, 0.0023424
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0088627, 0.0090875
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017242, 0.0017679
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0023006, 0.0022437
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002935, 0.0002862
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015502, 0.0015895
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0079577, 0.0077609

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054249, upper bound: 0.0051313
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051309, upper bound: 0.0053916
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004529, 0.0004349
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0024078, 0.0025076
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0056022, 0.0053793
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022668, 0.0023608
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0087945, 0.0091590
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017109, 0.0017818
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0023187, 0.0022265
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002958, 0.0002840
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015383, 0.0016020
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0080203, 0.0077011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052143, upper bound: 0.0053193
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049728, upper bound: 0.0056508
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004493, 0.0004383
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0024266, 0.0024879
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0055582, 0.0054213
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022845, 0.0023422
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0088631, 0.0090869
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017242, 0.0017678
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0023005, 0.0022438
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002934, 0.0002862
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015503, 0.0015895
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0079572, 0.0077612

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054260, upper bound: 0.0051314
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051314, upper bound: 0.0053916
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004529, 0.0004349
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0024079, 0.0025075
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0056020, 0.0053795
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022670, 0.0023607
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0087949, 0.0091585
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017110, 0.0017817
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0023186, 0.0022266
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002958, 0.0002840
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015384, 0.0016020
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0080199, 0.0077015

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052144, upper bound: 0.0053196
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049771, upper bound: 0.0056508
time: 0.84 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.38 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0056508, upper bound: 0.0049772
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0053196, upper bound: 0.0052144
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0053916, upper bound: 0.0051315
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0051314, upper bound: 0.0054260
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0056508, upper bound: 0.0049728
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0053193, upper bound: 0.0052143
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0053916, upper bound: 0.0051309
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0051313, upper bound: 0.0054249
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0055451, upper bound: 0.0050259
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0052691, upper bound: 0.0053318
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0053077, upper bound: 0.0051941
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0050916, upper bound: 0.0055576
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0055442, upper bound: 0.0050138
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0052645, upper bound: 0.0053289
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0053063, upper bound: 0.0051871
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0050905, upper bound: 0.0055480
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0055127, upper bound: 0.0050973
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0051443, upper bound: 0.0053134
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0052972, upper bound: 0.0052689
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0049894, upper bound: 0.0055447
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0055170, upper bound: 0.0050986
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0051509, upper bound: 0.0053149
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0053017, upper bound: 0.0052730
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0049957, upper bound: 0.0055459
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0054165, upper bound: 0.0051465
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0051081, upper bound: 0.0054188
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0052051, upper bound: 0.0053357
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0049595, upper bound: 0.0056662
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0054178, upper bound: 0.0051468
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0051090, upper bound: 0.0054190
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0052051, upper bound: 0.0053357
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0049614, upper bound: 0.0056644
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0056644, upper bound: 0.0049614
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0053357, upper bound: 0.0052051
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0054190, upper bound: 0.0051090
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0051468, upper bound: 0.0054178
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0056662, upper bound: 0.0049595
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0053357, upper bound: 0.0052051
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0054188, upper bound: 0.0051081
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0051465, upper bound: 0.0054165
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0055459, upper bound: 0.0049957
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0052730, upper bound: 0.0053017
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0053149, upper bound: 0.0051509
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0050986, upper bound: 0.0055170
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0055447, upper bound: 0.0049894
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0052689, upper bound: 0.0052972
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0053134, upper bound: 0.0051443
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0050973, upper bound: 0.0055127
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0055480, upper bound: 0.0050905
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0051871, upper bound: 0.0053063
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0053289, upper bound: 0.0052645
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0050138, upper bound: 0.0055442
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0055576, upper bound: 0.0050916
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0051941, upper bound: 0.0053077
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0053318, upper bound: 0.0052691
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0050259, upper bound: 0.0055451
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0054249, upper bound: 0.0051313
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0051309, upper bound: 0.0053916
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0052143, upper bound: 0.0053193
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0049728, upper bound: 0.0056508
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0054260, upper bound: 0.0051314
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0051314, upper bound: 0.0053916
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0052144, upper bound: 0.0053196
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 4, lower bound: -0.0049771, upper bound: 0.0056508

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0003896, 0.0004307
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023847, 0.0021571
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0048192, 0.0053277
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022451, 0.0020308
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0087102, 0.0078789
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016945, 0.0015328
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0019947, 0.0022051
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002544, 0.0002813
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015236, 0.0013781
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0068993, 0.0076273

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0056478, upper bound: 0.0048525
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054301, upper bound: 0.0049740
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004073, 0.0004142
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022933, 0.0022550
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050379, 0.0051235
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021590, 0.0021230
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0083763, 0.0082363
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016295, 0.0016023
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020851, 0.0021206
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002660, 0.0002705
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014651, 0.0014407
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0072123, 0.0073349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053169, upper bound: 0.0051541
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0050423, upper bound: 0.0052112
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0003930, 0.0004270
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023644, 0.0021758
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0048609, 0.0052824
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022260, 0.0020484
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0086361, 0.0079471
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016801, 0.0015460
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020119, 0.0021864
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002566, 0.0002789
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015106, 0.0013901
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0069590, 0.0075624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053885, upper bound: 0.0049251
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052892, upper bound: 0.0051284
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004108, 0.0004108
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022745, 0.0022747
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050819, 0.0050814
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021413, 0.0021415
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0083075, 0.0083082
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016161, 0.0016163
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021033, 0.0021032
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002683, 0.0002683
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014531, 0.0014532
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0072753, 0.0072747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051284, upper bound: 0.0052780
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049633, upper bound: 0.0054230
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0003896, 0.0004306
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023845, 0.0021570
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0048190, 0.0053272
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022449, 0.0020307
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0087094, 0.0078785
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016943, 0.0015327
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0019945, 0.0022049
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002544, 0.0002813
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015234, 0.0013781
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0068990, 0.0076266

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0056479, upper bound: 0.0048465
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054304, upper bound: 0.0049696
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004073, 0.0004142
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022932, 0.0022550
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050378, 0.0051232
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021589, 0.0021229
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0083758, 0.0082362
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016294, 0.0016023
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020851, 0.0021205
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002660, 0.0002705
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014651, 0.0014406
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0072123, 0.0073345

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053165, upper bound: 0.0051535
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0050418, upper bound: 0.0052111
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0003929, 0.0004270
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023642, 0.0021757
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0048607, 0.0052819
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022258, 0.0020483
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0086353, 0.0079467
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016799, 0.0015459
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020118, 0.0021862
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002566, 0.0002789
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015105, 0.0013900
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0069587, 0.0075617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053886, upper bound: 0.0049222
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052892, upper bound: 0.0051279
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004108, 0.0004108
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022743, 0.0022746
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050818, 0.0050811
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021412, 0.0021415
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0083070, 0.0083081
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016160, 0.0016163
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021033, 0.0021030
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002683, 0.0002683
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014530, 0.0014532
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0072752, 0.0072742

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051282, upper bound: 0.0052750
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049618, upper bound: 0.0054219
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0003903, 0.0004300
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023810, 0.0021611
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0048282, 0.0053193
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022416, 0.0020346
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0086965, 0.0078936
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016918, 0.0015356
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0019984, 0.0022016
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002549, 0.0002808
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015212, 0.0013807
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0069122, 0.0076153

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055422, upper bound: 0.0048717
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053635, upper bound: 0.0050227
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004082, 0.0004136
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022903, 0.0022602
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050494, 0.0051167
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021562, 0.0021278
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0083652, 0.0082552
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016274, 0.0016060
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020899, 0.0021178
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002666, 0.0002701
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014632, 0.0014440
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0072289, 0.0073252

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052662, upper bound: 0.0052479
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0050359, upper bound: 0.0053286
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0003937, 0.0004263
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023606, 0.0021801
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0048706, 0.0052738
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022224, 0.0020525
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0086221, 0.0079628
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016773, 0.0015491
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020159, 0.0021828
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002571, 0.0002784
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015081, 0.0013928
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0069728, 0.0075501

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053047, upper bound: 0.0049376
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052220, upper bound: 0.0051912
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004117, 0.0004103
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022716, 0.0022797
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050930, 0.0050750
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021386, 0.0021462
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0082970, 0.0083265
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016141, 0.0016198
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021080, 0.0021005
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002689, 0.0002679
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014513, 0.0014564
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0072913, 0.0072655

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0050885, upper bound: 0.0053656
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049539, upper bound: 0.0055547
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0003903, 0.0004300
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023807, 0.0021610
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0048280, 0.0053188
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022414, 0.0020345
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0086957, 0.0078932
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016917, 0.0015355
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0019983, 0.0022014
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002549, 0.0002808
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015210, 0.0013806
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0069119, 0.0076146

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055413, upper bound: 0.0048683
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053633, upper bound: 0.0050107
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004082, 0.0004136
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022901, 0.0022601
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050494, 0.0051164
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021561, 0.0021278
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0083647, 0.0082552
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016273, 0.0016060
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020899, 0.0021177
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002666, 0.0002701
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014631, 0.0014440
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0072288, 0.0073248

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052617, upper bound: 0.0052471
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0050357, upper bound: 0.0053257
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0003937, 0.0004263
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023604, 0.0021800
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0048704, 0.0052735
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022222, 0.0020524
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0086215, 0.0079625
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016772, 0.0015490
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020158, 0.0021827
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002571, 0.0002784
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015080, 0.0013928
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0069726, 0.0075496

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053032, upper bound: 0.0049343
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052210, upper bound: 0.0051841
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004117, 0.0004102
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022714, 0.0022797
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050930, 0.0050746
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021385, 0.0021462
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0082964, 0.0083264
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016140, 0.0016198
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021080, 0.0021004
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002689, 0.0002679
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014512, 0.0014564
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0072913, 0.0072650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0050874, upper bound: 0.0053641
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049523, upper bound: 0.0055449
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004032, 0.0004123
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022827, 0.0022323
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0049871, 0.0050998
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021491, 0.0021016
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0083376, 0.0081534
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016220, 0.0015862
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020641, 0.0021108
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002633, 0.0002693
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014584, 0.0014262
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0071397, 0.0073011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055097, upper bound: 0.0049630
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053236, upper bound: 0.0050943
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004208, 0.0003943
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0021835, 0.0023301
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0052058, 0.0048781
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0020557, 0.0021937
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0079752, 0.0085108
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0015515, 0.0016557
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021546, 0.0020190
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002748, 0.0002575
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0013950, 0.0014887
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0074527, 0.0069836

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051414, upper bound: 0.0052322
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049053, upper bound: 0.0053104
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004065, 0.0004086
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022622, 0.0022509
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050288, 0.0050540
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021298, 0.0021192
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0082627, 0.0082215
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016074, 0.0015994
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020814, 0.0020918
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002655, 0.0002668
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014453, 0.0014381
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0071994, 0.0072355

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052940, upper bound: 0.0050439
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051992, upper bound: 0.0052660
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004244, 0.0003908
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0021639, 0.0023498
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0052497, 0.0048343
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0020372, 0.0022123
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0079036, 0.0085827
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0015376, 0.0016697
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021728, 0.0020009
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002772, 0.0002552
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0013825, 0.0015013
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0075157, 0.0069209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049862, upper bound: 0.0053679
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0048433, upper bound: 0.0055419
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004031, 0.0004123
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022827, 0.0022322
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0049869, 0.0050998
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021491, 0.0021015
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0083376, 0.0081529
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016220, 0.0015861
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020640, 0.0021108
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002633, 0.0002693
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014584, 0.0014261
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0071393, 0.0073010

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055141, upper bound: 0.0049640
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053284, upper bound: 0.0050955
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004208, 0.0003944
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0021836, 0.0023301
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0052057, 0.0048784
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0020558, 0.0021937
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0079756, 0.0085107
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0015516, 0.0016557
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021546, 0.0020191
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002748, 0.0002576
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0013951, 0.0014887
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0074526, 0.0069840

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051480, upper bound: 0.0052332
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049108, upper bound: 0.0053119
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004065, 0.0004086
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022622, 0.0022508
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050286, 0.0050540
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021297, 0.0021191
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0082626, 0.0082212
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016074, 0.0015993
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020813, 0.0020918
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002655, 0.0002668
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014453, 0.0014380
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0071991, 0.0072354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052985, upper bound: 0.0050449
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052027, upper bound: 0.0052701
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004244, 0.0003908
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0021640, 0.0023498
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0052497, 0.0048347
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0020373, 0.0022122
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0079041, 0.0085826
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0015377, 0.0016697
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021728, 0.0020010
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002772, 0.0002553
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0013826, 0.0015012
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0075156, 0.0069214

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049925, upper bound: 0.0053679
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0048471, upper bound: 0.0055429
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004039, 0.0004115
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022784, 0.0022363
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0049961, 0.0050901
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021450, 0.0021054
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0083217, 0.0081681
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016189, 0.0015890
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020679, 0.0021068
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002638, 0.0002687
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014556, 0.0014287
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0071526, 0.0072871

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054136, upper bound: 0.0049763
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052596, upper bound: 0.0051435
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004218, 0.0003936
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0021795, 0.0023353
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0052173, 0.0048693
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0020519, 0.0021986
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0079607, 0.0085297
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0015487, 0.0016594
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021594, 0.0020154
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002755, 0.0002571
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0013925, 0.0014920
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0074692, 0.0069710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051052, upper bound: 0.0053152
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049001, upper bound: 0.0054158
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004073, 0.0004078
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022577, 0.0022552
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050385, 0.0050440
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021255, 0.0021232
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0082463, 0.0082373
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016042, 0.0016025
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020854, 0.0020877
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002660, 0.0002663
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014424, 0.0014408
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0072132, 0.0072210

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052020, upper bound: 0.0050542
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051286, upper bound: 0.0053329
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004253, 0.0003901
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0021599, 0.0023548
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0052609, 0.0048254
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0020334, 0.0022170
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0078890, 0.0086010
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0015347, 0.0016732
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021775, 0.0019972
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002778, 0.0002548
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0013799, 0.0015044
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0075317, 0.0069082

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049563, upper bound: 0.0054543
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0048308, upper bound: 0.0056633
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004039, 0.0004115
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022783, 0.0022362
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0049959, 0.0050901
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021450, 0.0021053
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0083216, 0.0081677
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016189, 0.0015889
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020678, 0.0021067
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002638, 0.0002687
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014556, 0.0014287
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0071522, 0.0072871

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054149, upper bound: 0.0049767
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052610, upper bound: 0.0051438
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004218, 0.0003937
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0021797, 0.0023353
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0052173, 0.0048696
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0020521, 0.0021986
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0079612, 0.0085297
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0015488, 0.0016594
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021594, 0.0020155
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002755, 0.0002571
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0013925, 0.0014920
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0074692, 0.0069714

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051061, upper bound: 0.0053151
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049053, upper bound: 0.0054160
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004073, 0.0004078
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022577, 0.0022552
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050383, 0.0050440
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021255, 0.0021231
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0082463, 0.0082370
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016042, 0.0016024
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020853, 0.0020877
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002660, 0.0002663
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014424, 0.0014408
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0072129, 0.0072211

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052020, upper bound: 0.0050542
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051310, upper bound: 0.0053329
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004253, 0.0003901
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0021600, 0.0023548
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0052609, 0.0048258
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0020336, 0.0022170
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0078895, 0.0086009
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0015348, 0.0016732
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021775, 0.0019974
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002778, 0.0002548
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0013800, 0.0015044
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0075316, 0.0069087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049581, upper bound: 0.0054546
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0048339, upper bound: 0.0056615
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0003901, 0.0004289
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023748, 0.0021600
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0048258, 0.0053056
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022358, 0.0020336
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0086740, 0.0078895
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016874, 0.0015348
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0019974, 0.0021959
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002548, 0.0002801
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015172, 0.0013800
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0069087, 0.0075956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0056615, upper bound: 0.0048339
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054546, upper bound: 0.0049581
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004078, 0.0004116
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022792, 0.0022577
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050440, 0.0050919
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021457, 0.0021255
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0083246, 0.0082463
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016195, 0.0016042
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020877, 0.0021075
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002663, 0.0002688
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014561, 0.0014424
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0072211, 0.0072897

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053329, upper bound: 0.0051310
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0050542, upper bound: 0.0052020
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0003937, 0.0004251
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023538, 0.0021797
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0048696, 0.0052586
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022160, 0.0020521
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0085972, 0.0079612
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016725, 0.0015488
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020155, 0.0021765
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002571, 0.0002776
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015038, 0.0013925
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0069714, 0.0075283

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054160, upper bound: 0.0049053
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053152, upper bound: 0.0051061
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004115, 0.0004084
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022615, 0.0022783
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050901, 0.0050524
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021291, 0.0021450
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0082601, 0.0083216
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016069, 0.0016189
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021067, 0.0020912
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002687, 0.0002667
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014448, 0.0014556
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0072871, 0.0072332

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051438, upper bound: 0.0052610
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049767, upper bound: 0.0054149
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0003901, 0.0004289
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023747, 0.0021599
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0048254, 0.0053052
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022356, 0.0020334
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0086734, 0.0078890
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016873, 0.0015347
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0019972, 0.0021958
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002548, 0.0002801
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015171, 0.0013799
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0069082, 0.0075951

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0056633, upper bound: 0.0048308
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054543, upper bound: 0.0049563
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004078, 0.0004116
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022790, 0.0022577
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050439, 0.0050916
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021456, 0.0021255
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0083242, 0.0082463
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016194, 0.0016042
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020877, 0.0021074
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002663, 0.0002688
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014560, 0.0014424
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0072210, 0.0072893

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053329, upper bound: 0.0051286
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0050542, upper bound: 0.0052020
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0003936, 0.0004251
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023536, 0.0021795
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0048693, 0.0052582
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022158, 0.0020519
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0085965, 0.0079607
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016724, 0.0015487
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020154, 0.0021763
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002571, 0.0002776
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015037, 0.0013925
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0069710, 0.0075278

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054158, upper bound: 0.0049001
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053151, upper bound: 0.0051052
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004115, 0.0004084
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022613, 0.0022784
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050901, 0.0050521
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021290, 0.0021450
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0082596, 0.0083217
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016068, 0.0016189
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021068, 0.0020910
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002687, 0.0002667
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014447, 0.0014556
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0072871, 0.0072327

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051435, upper bound: 0.0052596
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049763, upper bound: 0.0054136
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0003908, 0.0004282
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023708, 0.0021640
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0048347, 0.0052966
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022320, 0.0020373
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0086592, 0.0079041
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016846, 0.0015377
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020010, 0.0021922
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002553, 0.0002796
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015146, 0.0013826
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0069214, 0.0075827

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055429, upper bound: 0.0048471
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053679, upper bound: 0.0049925
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004086, 0.0004110
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022758, 0.0022622
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050540, 0.0050845
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021426, 0.0021297
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0083125, 0.0082626
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016171, 0.0016074
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020918, 0.0021044
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002668, 0.0002684
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014540, 0.0014453
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0072354, 0.0072791

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052701, upper bound: 0.0052027
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0050449, upper bound: 0.0052985
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0003944, 0.0004243
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023496, 0.0021836
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0048784, 0.0052492
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022120, 0.0020558
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0085819, 0.0079756
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016695, 0.0015516
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020191, 0.0021726
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002576, 0.0002771
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015011, 0.0013951
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0069840, 0.0075149

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053119, upper bound: 0.0049108
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052332, upper bound: 0.0051480
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004123, 0.0004078
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022581, 0.0022827
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050998, 0.0050449
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021259, 0.0021491
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0082478, 0.0083376
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016045, 0.0016220
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021108, 0.0020880
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002693, 0.0002663
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014427, 0.0014584
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0073010, 0.0072224

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0050955, upper bound: 0.0053284
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049640, upper bound: 0.0055141
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0003908, 0.0004281
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023706, 0.0021639
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0048343, 0.0052962
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022318, 0.0020372
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0086587, 0.0079036
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016845, 0.0015376
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020009, 0.0021921
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002552, 0.0002796
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015145, 0.0013825
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0069209, 0.0075822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055419, upper bound: 0.0048433
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053679, upper bound: 0.0049862
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004086, 0.0004110
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022757, 0.0022622
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050540, 0.0050842
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021425, 0.0021298
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0083121, 0.0082627
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016170, 0.0016074
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020918, 0.0021043
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002668, 0.0002684
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014539, 0.0014453
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0072355, 0.0072787

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052660, upper bound: 0.0051992
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0050439, upper bound: 0.0052940
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0003943, 0.0004243
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023494, 0.0021835
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0048781, 0.0052488
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022119, 0.0020557
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0085812, 0.0079752
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016694, 0.0015515
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020190, 0.0021725
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002575, 0.0002771
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015010, 0.0013950
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0069836, 0.0075144

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053104, upper bound: 0.0049053
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052322, upper bound: 0.0051414
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004123, 0.0004078
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022580, 0.0022827
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050998, 0.0050445
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021258, 0.0021491
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0082472, 0.0083376
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016044, 0.0016220
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021108, 0.0020879
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002693, 0.0002663
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014426, 0.0014584
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0073010, 0.0072218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0050943, upper bound: 0.0053236
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049630, upper bound: 0.0055097
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004037, 0.0004117
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022797, 0.0022352
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0049937, 0.0050930
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021462, 0.0021043
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0083264, 0.0081640
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016198, 0.0015882
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020668, 0.0021080
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002636, 0.0002689
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014564, 0.0014280
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0071490, 0.0072913

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055449, upper bound: 0.0049523
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053641, upper bound: 0.0050874
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004213, 0.0003937
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0021800, 0.0023329
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0052119, 0.0048704
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0020524, 0.0021963
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0079625, 0.0085208
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0015490, 0.0016576
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021572, 0.0020158
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002752, 0.0002571
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0013928, 0.0014904
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0074614, 0.0069726

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051841, upper bound: 0.0052210
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049343, upper bound: 0.0053032
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004072, 0.0004082
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022601, 0.0022548
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050375, 0.0050494
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021278, 0.0021228
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0082552, 0.0082357
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016060, 0.0016022
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020850, 0.0020899
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002660, 0.0002666
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014440, 0.0014406
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0072118, 0.0072288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053257, upper bound: 0.0050357
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052471, upper bound: 0.0052617
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004251, 0.0003903
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0021610, 0.0023535
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0052580, 0.0048280
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0020345, 0.0022157
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0078932, 0.0085961
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0015355, 0.0016723
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021762, 0.0019983
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002776, 0.0002549
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0013806, 0.0015036
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0075274, 0.0069119

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0050107, upper bound: 0.0053633
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0048683, upper bound: 0.0055413
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004037, 0.0004117
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022797, 0.0022350
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0049933, 0.0050930
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021462, 0.0021042
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0083265, 0.0081635
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016198, 0.0015881
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020667, 0.0021080
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002636, 0.0002689
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014564, 0.0014279
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0071485, 0.0072913

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055547, upper bound: 0.0049539
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053656, upper bound: 0.0050885
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004213, 0.0003937
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0021801, 0.0023328
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0052118, 0.0048706
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0020525, 0.0021963
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0079628, 0.0085207
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0015491, 0.0016576
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021572, 0.0020159
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002752, 0.0002571
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0013928, 0.0014904
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0074614, 0.0069728

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051912, upper bound: 0.0052220
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049376, upper bound: 0.0053047
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004072, 0.0004082
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022602, 0.0022547
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050372, 0.0050494
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021278, 0.0021227
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0082552, 0.0082352
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016060, 0.0016021
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020849, 0.0020899
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002659, 0.0002666
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014440, 0.0014405
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0072114, 0.0072289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053286, upper bound: 0.0050359
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052479, upper bound: 0.0052662
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004251, 0.0003903
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0021611, 0.0023535
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0052580, 0.0048282
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0020346, 0.0022157
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0078936, 0.0085962
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0015356, 0.0016723
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021763, 0.0019984
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002776, 0.0002549
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0013807, 0.0015036
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0075275, 0.0069122

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0050227, upper bound: 0.0053635
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0048717, upper bound: 0.0055422
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004044, 0.0004108
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022746, 0.0022392
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050026, 0.0050818
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021415, 0.0021081
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0083081, 0.0081786
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016163, 0.0015911
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020705, 0.0021033
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002641, 0.0002683
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014532, 0.0014306
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0071618, 0.0072752

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054219, upper bound: 0.0049618
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052750, upper bound: 0.0051282
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004221, 0.0003929
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0021757, 0.0023373
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0052218, 0.0048607
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0020483, 0.0022005
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0079467, 0.0085371
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0015459, 0.0016608
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021613, 0.0020118
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002757, 0.0002566
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0013900, 0.0014933
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0074757, 0.0069587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051279, upper bound: 0.0052892
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049222, upper bound: 0.0053886
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004079, 0.0004073
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022550, 0.0022587
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050463, 0.0050378
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021229, 0.0021265
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0082362, 0.0082501
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016023, 0.0016050
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020886, 0.0020851
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002664, 0.0002660
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014406, 0.0014431
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0072244, 0.0072123

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052112, upper bound: 0.0050418
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051535, upper bound: 0.0053165
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004258, 0.0003896
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0021570, 0.0023579
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0052677, 0.0048190
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0020307, 0.0022198
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0078785, 0.0086121
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0015327, 0.0016754
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021803, 0.0019945
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002781, 0.0002544
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0013781, 0.0015064
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0075414, 0.0068990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049696, upper bound: 0.0054304
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0048465, upper bound: 0.0056479
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004044, 0.0004108
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022747, 0.0022390
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050022, 0.0050819
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021415, 0.0021079
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0083082, 0.0081780
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016163, 0.0015910
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020704, 0.0021033
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002641, 0.0002683
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014532, 0.0014305
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0071613, 0.0072753

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054230, upper bound: 0.0049633
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052780, upper bound: 0.0051284
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004221, 0.0003930
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0021758, 0.0023374
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0052219, 0.0048609
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0020484, 0.0022005
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0079471, 0.0085372
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0015460, 0.0016608
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021613, 0.0020119
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002757, 0.0002566
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0013901, 0.0014933
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0074758, 0.0069590

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051284, upper bound: 0.0052892
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049251, upper bound: 0.0053885
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004079, 0.0004073
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022550, 0.0022586
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050460, 0.0050379
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021230, 0.0021264
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0082363, 0.0082496
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016023, 0.0016049
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020885, 0.0020851
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002664, 0.0002660
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014407, 0.0014430
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0072240, 0.0072123

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052112, upper bound: 0.0050423
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051541, upper bound: 0.0053169
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004258, 0.0003896
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0021571, 0.0023579
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0052677, 0.0048192
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0020308, 0.0022198
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0078789, 0.0086121
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0015328, 0.0016754
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021803, 0.0019947
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002781, 0.0002544
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0013781, 0.0015064
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0075414, 0.0068993

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049740, upper bound: 0.0054301
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0048525, upper bound: 0.0056478
time: 0.84 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.30 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0056478, upper bound: 0.0048525
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0054301, upper bound: 0.0049740
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0053169, upper bound: 0.0051541
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0050423, upper bound: 0.0052112
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0053885, upper bound: 0.0049251
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0052892, upper bound: 0.0051284
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0051284, upper bound: 0.0052780
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0049633, upper bound: 0.0054230
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0056479, upper bound: 0.0048465
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0054304, upper bound: 0.0049696
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0053165, upper bound: 0.0051535
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0050418, upper bound: 0.0052111
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0053886, upper bound: 0.0049222
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0052892, upper bound: 0.0051279
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0051282, upper bound: 0.0052750
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0049618, upper bound: 0.0054219
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0055422, upper bound: 0.0048717
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0053635, upper bound: 0.0050227
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0052662, upper bound: 0.0052479
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0050359, upper bound: 0.0053286
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0053047, upper bound: 0.0049376
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0052220, upper bound: 0.0051912
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0050885, upper bound: 0.0053656
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0049539, upper bound: 0.0055547
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0055413, upper bound: 0.0048683
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0053633, upper bound: 0.0050107
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0052617, upper bound: 0.0052471
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0050357, upper bound: 0.0053257
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0053032, upper bound: 0.0049343
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0052210, upper bound: 0.0051841
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0050874, upper bound: 0.0053641
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0049523, upper bound: 0.0055449
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0055097, upper bound: 0.0049630
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0053236, upper bound: 0.0050943
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0051414, upper bound: 0.0052322
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0049053, upper bound: 0.0053104
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0052940, upper bound: 0.0050439
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0051992, upper bound: 0.0052660
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0049862, upper bound: 0.0053679
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0048433, upper bound: 0.0055419
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0055141, upper bound: 0.0049640
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0053284, upper bound: 0.0050955
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0051480, upper bound: 0.0052332
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0049108, upper bound: 0.0053119
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0052985, upper bound: 0.0050449
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0052027, upper bound: 0.0052701
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0049925, upper bound: 0.0053679
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0048471, upper bound: 0.0055429
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0054136, upper bound: 0.0049763
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0052596, upper bound: 0.0051435
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0051052, upper bound: 0.0053152
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0049001, upper bound: 0.0054158
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0052020, upper bound: 0.0050542
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0051286, upper bound: 0.0053329
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0049563, upper bound: 0.0054543
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0048308, upper bound: 0.0056633
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0054149, upper bound: 0.0049767
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0052610, upper bound: 0.0051438
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0051061, upper bound: 0.0053151
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0049053, upper bound: 0.0054160
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0052020, upper bound: 0.0050542
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0051310, upper bound: 0.0053329
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0049581, upper bound: 0.0054546
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0048339, upper bound: 0.0056615
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0056615, upper bound: 0.0048339
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0054546, upper bound: 0.0049581
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0053329, upper bound: 0.0051310
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0050542, upper bound: 0.0052020
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0054160, upper bound: 0.0049053
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0053152, upper bound: 0.0051061
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0051438, upper bound: 0.0052610
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0049767, upper bound: 0.0054149
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0056633, upper bound: 0.0048308
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0054543, upper bound: 0.0049563
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0053329, upper bound: 0.0051286
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0050542, upper bound: 0.0052020
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0054158, upper bound: 0.0049001
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0053151, upper bound: 0.0051052
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0051435, upper bound: 0.0052596
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0049763, upper bound: 0.0054136
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0055429, upper bound: 0.0048471
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0053679, upper bound: 0.0049925
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0052701, upper bound: 0.0052027
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0050449, upper bound: 0.0052985
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0053119, upper bound: 0.0049108
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0052332, upper bound: 0.0051480
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0050955, upper bound: 0.0053284
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0049640, upper bound: 0.0055141
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0055419, upper bound: 0.0048433
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0053679, upper bound: 0.0049862
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0052660, upper bound: 0.0051992
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0050439, upper bound: 0.0052940
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0053104, upper bound: 0.0049053
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0052322, upper bound: 0.0051414
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0050943, upper bound: 0.0053236
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0049630, upper bound: 0.0055097
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0055449, upper bound: 0.0049523
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0053641, upper bound: 0.0050874
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0051841, upper bound: 0.0052210
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0049343, upper bound: 0.0053032
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0053257, upper bound: 0.0050357
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0052471, upper bound: 0.0052617
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0050107, upper bound: 0.0053633
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0048683, upper bound: 0.0055413
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0055547, upper bound: 0.0049539
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0053656, upper bound: 0.0050885
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0051912, upper bound: 0.0052220
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0049376, upper bound: 0.0053047
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0053286, upper bound: 0.0050359
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0052479, upper bound: 0.0052662
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0050227, upper bound: 0.0053635
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0048717, upper bound: 0.0055422
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0054219, upper bound: 0.0049618
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0052750, upper bound: 0.0051282
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0051279, upper bound: 0.0052892
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0049222, upper bound: 0.0053886
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0052112, upper bound: 0.0050418
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0051535, upper bound: 0.0053165
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0049696, upper bound: 0.0054304
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0048465, upper bound: 0.0056479
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0054230, upper bound: 0.0049633
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0052780, upper bound: 0.0051284
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0051284, upper bound: 0.0052892
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0049251, upper bound: 0.0053885
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0052112, upper bound: 0.0050423
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0051541, upper bound: 0.0053169
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0049740, upper bound: 0.0054301
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -0.0048525, upper bound: 0.0056478

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0003926, 0.0004368
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0024187, 0.0021737
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0048562, 0.0054037
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022771, 0.0020464
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0088344, 0.0079394
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017186, 0.0015445
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020100, 0.0022365
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002564, 0.0002853
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015453, 0.0013887
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0069523, 0.0077360

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0056213, upper bound: 0.0048212
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055666, upper bound: 0.0048254
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0003966, 0.0004337
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0024013, 0.0021959
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0049059, 0.0053647
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022607, 0.0020674
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0087707, 0.0080206
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017063, 0.0015603
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020305, 0.0022204
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002590, 0.0002832
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015341, 0.0014029
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0070234, 0.0076803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054048, upper bound: 0.0049303
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053873, upper bound: 0.0049461
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004103, 0.0004212
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023321, 0.0022716
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050749, 0.0052101
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021955, 0.0021386
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0085179, 0.0082968
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016571, 0.0016141
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021005, 0.0021564
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002679, 0.0002751
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014899, 0.0014513
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0072653, 0.0074589

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052907, upper bound: 0.0051232
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052409, upper bound: 0.0051271
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004135, 0.0004172
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023099, 0.0022895
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0051150, 0.0051605
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021746, 0.0021555
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0084368, 0.0083623
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016413, 0.0016268
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021170, 0.0021359
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002700, 0.0002725
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014757, 0.0014627
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0073227, 0.0073879

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0050181, upper bound: 0.0051712
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0050057, upper bound: 0.0051838
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0003959, 0.0004332
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023986, 0.0021924
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0048980, 0.0053587
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022582, 0.0020640
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0087608, 0.0080076
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017043, 0.0015578
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020272, 0.0022179
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002586, 0.0002829
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015324, 0.0014007
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0070120, 0.0076716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053609, upper bound: 0.0048954
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053365, upper bound: 0.0048986
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004000, 0.0004300
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023810, 0.0022149
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0049483, 0.0053194
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022416, 0.0020852
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0086966, 0.0080899
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016918, 0.0015738
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020481, 0.0022017
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002613, 0.0002808
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015212, 0.0014151
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0070841, 0.0076154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052634, upper bound: 0.0050641
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052519, upper bound: 0.0050999
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004138, 0.0004177
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023130, 0.0022912
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0051189, 0.0051676
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021776, 0.0021571
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0084484, 0.0083687
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016435, 0.0016281
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021187, 0.0021388
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002703, 0.0002728
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014778, 0.0014638
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0073283, 0.0073980

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051032, upper bound: 0.0052403
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0050818, upper bound: 0.0052504
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004170, 0.0004138
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022910, 0.0023090
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0051586, 0.0051184
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021569, 0.0021738
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0083681, 0.0084337
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016279, 0.0016407
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021351, 0.0021185
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002724, 0.0002702
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014637, 0.0014752
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0073852, 0.0073277

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049377, upper bound: 0.0053544
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049340, upper bound: 0.0053953
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0003926, 0.0004368
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0024185, 0.0021736
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0048560, 0.0054033
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022769, 0.0020463
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0088337, 0.0079390
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017185, 0.0015444
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020099, 0.0022364
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002564, 0.0002853
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015452, 0.0013887
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0069520, 0.0077354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0056214, upper bound: 0.0048163
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055663, upper bound: 0.0048194
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0003966, 0.0004336
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0024011, 0.0021958
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0049057, 0.0053642
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022605, 0.0020673
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0087699, 0.0080202
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017061, 0.0015602
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020304, 0.0022202
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002590, 0.0002832
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015340, 0.0014029
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0070231, 0.0076796

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054053, upper bound: 0.0049264
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053868, upper bound: 0.0049416
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004102, 0.0004212
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023319, 0.0022715
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050748, 0.0052098
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021954, 0.0021385
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0085174, 0.0082967
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016570, 0.0016140
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021004, 0.0021563
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002679, 0.0002751
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014898, 0.0014512
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0072653, 0.0074584

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052905, upper bound: 0.0051216
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052409, upper bound: 0.0051264
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004135, 0.0004171
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023097, 0.0022894
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0051149, 0.0051602
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021745, 0.0021554
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0084363, 0.0083622
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016412, 0.0016268
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021170, 0.0021358
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002700, 0.0002724
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014757, 0.0014627
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0073226, 0.0073875

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0050178, upper bound: 0.0051712
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0050055, upper bound: 0.0051837
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0003959, 0.0004332
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023984, 0.0021923
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0048977, 0.0053582
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022580, 0.0020639
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0087600, 0.0080072
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017042, 0.0015577
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020271, 0.0022177
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002586, 0.0002829
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015323, 0.0014006
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0070117, 0.0076710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053611, upper bound: 0.0048895
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053365, upper bound: 0.0048959
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004000, 0.0004300
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023808, 0.0022148
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0049481, 0.0053190
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022414, 0.0020852
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0086958, 0.0080896
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016917, 0.0015738
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020480, 0.0022015
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002612, 0.0002808
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015210, 0.0014150
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0070839, 0.0076147

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052634, upper bound: 0.0050599
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052520, upper bound: 0.0050994
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004138, 0.0004177
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023129, 0.0022912
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0051188, 0.0051672
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021775, 0.0021571
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0084478, 0.0083686
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016434, 0.0016280
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021186, 0.0021387
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002703, 0.0002728
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014777, 0.0014638
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0073282, 0.0073975

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051031, upper bound: 0.0052359
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0050819, upper bound: 0.0052477
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004170, 0.0004137
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022909, 0.0023089
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0051584, 0.0051181
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021568, 0.0021738
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0083675, 0.0084334
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016278, 0.0016406
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021350, 0.0021184
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002723, 0.0002702
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014636, 0.0014751
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0073849, 0.0073272

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049366, upper bound: 0.0053531
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049334, upper bound: 0.0053937
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0003933, 0.0004362
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0024152, 0.0021777
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0048653, 0.0053958
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022738, 0.0020502
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0088215, 0.0079541
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017161, 0.0015474
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020137, 0.0022333
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002569, 0.0002849
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015430, 0.0013913
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0069652, 0.0077248

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055165, upper bound: 0.0048435
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054476, upper bound: 0.0048448
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0003973, 0.0004330
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023975, 0.0021999
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0049147, 0.0053564
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022572, 0.0020711
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0087570, 0.0080350
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017036, 0.0015631
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020342, 0.0022170
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002595, 0.0002828
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015317, 0.0014055
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0070361, 0.0076683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053389, upper bound: 0.0049797
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053150, upper bound: 0.0049949
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004112, 0.0004207
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023294, 0.0022767
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050864, 0.0052041
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021930, 0.0021434
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0085081, 0.0083157
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016552, 0.0016177
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021052, 0.0021540
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002685, 0.0002748
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014882, 0.0014546
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0072819, 0.0074503

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052406, upper bound: 0.0052206
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051919, upper bound: 0.0052211
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004144, 0.0004166
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023068, 0.0022947
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0051266, 0.0051537
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021718, 0.0021604
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0084257, 0.0083814
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016391, 0.0016305
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021219, 0.0021331
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002707, 0.0002721
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014738, 0.0014660
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0073393, 0.0073782

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0050119, upper bound: 0.0052904
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0050001, upper bound: 0.0052999
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0003967, 0.0004326
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023951, 0.0021967
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0049076, 0.0053508
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022549, 0.0020681
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0087480, 0.0080233
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017018, 0.0015609
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020312, 0.0022147
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002591, 0.0002825
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015302, 0.0014034
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0070258, 0.0076604

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052791, upper bound: 0.0049101
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052552, upper bound: 0.0049110
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004008, 0.0004293
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023772, 0.0022192
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0049579, 0.0053108
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022380, 0.0020893
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0086826, 0.0081056
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016891, 0.0015769
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020521, 0.0021981
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002618, 0.0002804
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015187, 0.0014178
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0070979, 0.0076031

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051967, upper bound: 0.0051308
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0051850, upper bound: 0.0051627
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004147, 0.0004173
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023104, 0.0022962
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0051300, 0.0051618
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021752, 0.0021618
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0084389, 0.0083870
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016417, 0.0016316
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021233, 0.0021364
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002708, 0.0002725
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014761, 0.0014670
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0073443, 0.0073897

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0050637, upper bound: 0.0053293
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0050365, upper bound: 0.0053381
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004179, 0.0004133
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022882, 0.0023139
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0051695, 0.0051120
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021542, 0.0021785
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0083575, 0.0084516
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016259, 0.0016442
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021396, 0.0021158
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002729, 0.0002699
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014619, 0.0014783
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0074008, 0.0073185

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049287, upper bound: 0.0054890
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049197, upper bound: 0.0055257
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0003933, 0.0004362
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0024150, 0.0021776
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0048650, 0.0053954
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022736, 0.0020501
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0088208, 0.0079537
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017160, 0.0015473
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020136, 0.0022331
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002569, 0.0002849
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015429, 0.0013912
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0069649, 0.0077241

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055158, upper bound: 0.0048392
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054448, upper bound: 0.0048411
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0003973, 0.0004330
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023973, 0.0021997
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0049145, 0.0053559
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022570, 0.0020710
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0087562, 0.0080346
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017034, 0.0015630
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020341, 0.0022168
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002595, 0.0002828
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015316, 0.0014054
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0070357, 0.0076676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053387, upper bound: 0.0049681
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053141, upper bound: 0.0049824
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004112, 0.0004207
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023292, 0.0022767
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050864, 0.0052038
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021929, 0.0021434
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0085076, 0.0083157
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016551, 0.0016177
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021052, 0.0021538
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002685, 0.0002747
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014881, 0.0014545
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0072818, 0.0074499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052361, upper bound: 0.0052198
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051874, upper bound: 0.0052203
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004144, 0.0004166
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023067, 0.0022946
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0051265, 0.0051534
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021717, 0.0021603
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0084253, 0.0083812
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016390, 0.0016305
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021218, 0.0021330
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002707, 0.0002721
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014737, 0.0014660
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0073392, 0.0073778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0050117, upper bound: 0.0052891
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0050000, upper bound: 0.0052972
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0003967, 0.0004325
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023948, 0.0021966
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0049074, 0.0053503
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022546, 0.0020680
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0087470, 0.0080230
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0017016, 0.0015608
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020311, 0.0022144
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002591, 0.0002825
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015300, 0.0014034
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0070255, 0.0076596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052778, upper bound: 0.0049053
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052516, upper bound: 0.0049075
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004008, 0.0004293
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023770, 0.0022191
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0049577, 0.0053105
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0022378, 0.0020892
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0086820, 0.0081052
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016890, 0.0015768
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020520, 0.0021980
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002617, 0.0002804
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0015186, 0.0014177
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0070975, 0.0076026

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051957, upper bound: 0.0051162
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0051847, upper bound: 0.0051550
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004147, 0.0004173
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023103, 0.0022962
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0051300, 0.0051615
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021751, 0.0021618
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0084384, 0.0083870
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016416, 0.0016316
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021233, 0.0021363
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002708, 0.0002725
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014760, 0.0014670
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0073443, 0.0073893

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0050625, upper bound: 0.0053265
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0050339, upper bound: 0.0053366
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004179, 0.0004132
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022880, 0.0023139
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0051694, 0.0051116
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021541, 0.0021784
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0083569, 0.0084514
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016258, 0.0016441
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021396, 0.0021157
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002729, 0.0002699
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014618, 0.0014783
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0074007, 0.0073180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049274, upper bound: 0.0054808
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049191, upper bound: 0.0055160
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004057, 0.0004186
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023180, 0.0022464
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050186, 0.0051786
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021823, 0.0021149
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0084664, 0.0082048
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016471, 0.0015962
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020772, 0.0021434
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002650, 0.0002734
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014809, 0.0014352
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0071848, 0.0074138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054806, upper bound: 0.0049309
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054320, upper bound: 0.0049379
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004097, 0.0004153
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022993, 0.0022686
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050683, 0.0051369
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021647, 0.0021358
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0083981, 0.0082860
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016338, 0.0016120
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020977, 0.0021261
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002676, 0.0002712
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014690, 0.0014494
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0072559, 0.0073540

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052959, upper bound: 0.0050429
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052838, upper bound: 0.0050688
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004234, 0.0004014
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022226, 0.0023442
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0052373, 0.0049654
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0020924, 0.0022070
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0081179, 0.0085623
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0015793, 0.0016657
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021677, 0.0020552
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002765, 0.0002622
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014200, 0.0014977
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0074978, 0.0071086

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051126, upper bound: 0.0051993
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0050677, upper bound: 0.0052068
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004266, 0.0003973
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022000, 0.0023622
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0052773, 0.0049151
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0020713, 0.0022239
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0080357, 0.0086278
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0015633, 0.0016784
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021843, 0.0020344
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002786, 0.0002595
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014056, 0.0015091
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0075551, 0.0070366

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0048783, upper bound: 0.0052613
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0048673, upper bound: 0.0052848
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004091, 0.0004148
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022967, 0.0022650
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050603, 0.0051311
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021623, 0.0021324
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0083888, 0.0082730
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016320, 0.0016094
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020944, 0.0021238
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002672, 0.0002709
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014673, 0.0014471
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0072445, 0.0073459

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052652, upper bound: 0.0050120
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052484, upper bound: 0.0050200
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004131, 0.0004116
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022788, 0.0022876
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0051107, 0.0050910
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021454, 0.0021537
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0083232, 0.0083553
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016192, 0.0016254
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021153, 0.0021072
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002698, 0.0002688
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014559, 0.0014615
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0073166, 0.0072885

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051719, upper bound: 0.0052024
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051677, upper bound: 0.0052407
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004269, 0.0003978
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022024, 0.0023639
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0052812, 0.0049205
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0020735, 0.0022255
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0080444, 0.0086342
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0015650, 0.0016797
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021859, 0.0020366
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002788, 0.0002598
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014071, 0.0015103
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0075608, 0.0070443

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049581, upper bound: 0.0053239
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049372, upper bound: 0.0053432
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004301, 0.0003938
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0021804, 0.0023817
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0053210, 0.0048713
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0020528, 0.0022423
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0079641, 0.0086992
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0015493, 0.0016923
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022023, 0.0020162
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002809, 0.0002572
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0013930, 0.0015216
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0076176, 0.0069739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0048159, upper bound: 0.0054491
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0048052, upper bound: 0.0055161
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004057, 0.0004186
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023179, 0.0022463
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050184, 0.0051785
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021822, 0.0021148
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0084663, 0.0082044
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016470, 0.0015961
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020771, 0.0021434
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002650, 0.0002734
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014809, 0.0014351
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0071844, 0.0074137

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054852, upper bound: 0.0049323
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054386, upper bound: 0.0049388
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004097, 0.0004153
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022993, 0.0022685
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050680, 0.0051369
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021647, 0.0021357
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0083981, 0.0082856
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016338, 0.0016119
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020976, 0.0021261
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002676, 0.0002712
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014690, 0.0014493
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0072555, 0.0073540

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053010, upper bound: 0.0050464
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052862, upper bound: 0.0050701
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004234, 0.0004014
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022227, 0.0023442
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0052372, 0.0049657
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0020926, 0.0022070
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0081184, 0.0085622
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0015794, 0.0016657
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021676, 0.0020553
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002765, 0.0002622
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014200, 0.0014977
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0074977, 0.0071091

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051193, upper bound: 0.0052012
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0050734, upper bound: 0.0052080
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004266, 0.0003974
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022002, 0.0023621
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0052772, 0.0049154
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0020714, 0.0022238
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0080361, 0.0086277
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0015633, 0.0016784
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021842, 0.0020345
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002786, 0.0002595
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014056, 0.0015091
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0075550, 0.0070370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0048838, upper bound: 0.0052647
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0048735, upper bound: 0.0052864
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004091, 0.0004148
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022967, 0.0022649
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050601, 0.0051311
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021623, 0.0021323
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0083887, 0.0082727
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016319, 0.0016094
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020943, 0.0021237
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002672, 0.0002709
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014673, 0.0014470
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0072442, 0.0073458

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052698, upper bound: 0.0050119
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052524, upper bound: 0.0050209
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004131, 0.0004116
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022787, 0.0022875
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0051105, 0.0050910
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021453, 0.0021536
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0083231, 0.0083551
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016192, 0.0016254
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021152, 0.0021071
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002698, 0.0002688
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014558, 0.0014614
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0073163, 0.0072884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051751, upper bound: 0.0052037
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051714, upper bound: 0.0052445
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004269, 0.0003978
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022026, 0.0023639
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0052812, 0.0049208
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0020737, 0.0022255
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0080450, 0.0086341
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0015651, 0.0016797
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021858, 0.0020367
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002788, 0.0002598
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014072, 0.0015102
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0075607, 0.0070448

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049646, upper bound: 0.0053239
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049472, upper bound: 0.0053432
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004301, 0.0003938
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0021806, 0.0023816
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0053208, 0.0048717
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0020529, 0.0022422
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0079646, 0.0086989
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0015494, 0.0016923
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0022022, 0.0020164
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002809, 0.0002572
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0013931, 0.0015216
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0076174, 0.0069744

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0048195, upper bound: 0.0054519
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0048117, upper bound: 0.0055170
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004064, 0.0004178
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0023133, 0.0022504
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050276, 0.0051683
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021779, 0.0021187
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0084495, 0.0082196
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016438, 0.0015990
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0020809, 0.0021391
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002654, 0.0002729
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014780, 0.0014377
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0071977, 0.0073990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053850, upper bound: 0.0049483
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053382, upper bound: 0.0049511
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004104, 0.0004145
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022949, 0.0022725
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0050771, 0.0051271
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0021606, 0.0021395
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0083823, 0.0083005
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0016307, 0.0016148
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021014, 0.0021221
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002681, 0.0002707
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014662, 0.0014519
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0072685, 0.0073401

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052328, upper bound: 0.0050979
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052185, upper bound: 0.0051178
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042520, -0.0036886, -0.0042520, -0.0036886, -0.0004243, 0.0004007
1: 0.0001719, 0.0032914, 0.0001719, 0.0032914, -0.0022185, 0.0023494
2: 0.0076128, 0.0145821, 0.0076128, 0.0145821, -0.0052488, 0.0049564
3: 0.0011894, 0.0041263, 0.0011894, 0.0041263, -0.0020887, 0.0022119
4: 1.0013648, 1.0127589, 1.0013648, 1.0127589, -0.0081032, 0.0085812
5: 0.0025069, 0.0047235, 0.0025069, 0.0047235, -0.0015764, 0.0016694
6: -0.0118899, -0.0090053, -0.0118899, -0.0090053, -0.0021725, 0.0020514
7: -0.0103200, -0.0099521, -0.0103200, -0.0099521, -0.0002771, 0.0002617
8: -0.0046321, -0.0026392, -0.0046321, -0.0026392, -0.0014174, 0.0015010
9: -0.0049587, 0.0050187, -0.0049587, 0.0050187, -0.0075143, 0.0070958

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0050766, upper bound: 0.0052829
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0050302, upper bound: 0.0052893
time: 0.93 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.53 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0056213, upper bound: 0.0048212
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0055666, upper bound: 0.0048254
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0054048, upper bound: 0.0049303
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0053873, upper bound: 0.0049461
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0052907, upper bound: 0.0051232
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0052409, upper bound: 0.0051271
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0050181, upper bound: 0.0051712
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0050057, upper bound: 0.0051838
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0053609, upper bound: 0.0048954
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0053365, upper bound: 0.0048986
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0052634, upper bound: 0.0050641
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0052519, upper bound: 0.0050999
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0051032, upper bound: 0.0052403
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0050818, upper bound: 0.0052504
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0049377, upper bound: 0.0053544
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0049340, upper bound: 0.0053953
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0056214, upper bound: 0.0048163
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0055663, upper bound: 0.0048194
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0054053, upper bound: 0.0049264
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0053868, upper bound: 0.0049416
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0052905, upper bound: 0.0051216
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0052409, upper bound: 0.0051264
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0050178, upper bound: 0.0051712
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0050055, upper bound: 0.0051837
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0053611, upper bound: 0.0048895
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0053365, upper bound: 0.0048959
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0052634, upper bound: 0.0050599
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0052520, upper bound: 0.0050994
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0051031, upper bound: 0.0052359
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0050819, upper bound: 0.0052477
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0049366, upper bound: 0.0053531
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0049334, upper bound: 0.0053937
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0055165, upper bound: 0.0048435
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0054476, upper bound: 0.0048448
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0053389, upper bound: 0.0049797
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0053150, upper bound: 0.0049949
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0052406, upper bound: 0.0052206
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0051919, upper bound: 0.0052211
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0050119, upper bound: 0.0052904
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0050001, upper bound: 0.0052999
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0052791, upper bound: 0.0049101
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0052552, upper bound: 0.0049110
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0051967, upper bound: 0.0051308
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0051850, upper bound: 0.0051627
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0050637, upper bound: 0.0053293
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0050365, upper bound: 0.0053381
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0049287, upper bound: 0.0054890
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0049197, upper bound: 0.0055257
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0055158, upper bound: 0.0048392
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0054448, upper bound: 0.0048411
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0053387, upper bound: 0.0049681
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0053141, upper bound: 0.0049824
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0052361, upper bound: 0.0052198
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0051874, upper bound: 0.0052203
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0050117, upper bound: 0.0052891
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0050000, upper bound: 0.0052972
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0052778, upper bound: 0.0049053
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0052516, upper bound: 0.0049075
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0051957, upper bound: 0.0051162
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0051847, upper bound: 0.0051550
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0050625, upper bound: 0.0053265
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0050339, upper bound: 0.0053366
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0049274, upper bound: 0.0054808
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0049191, upper bound: 0.0055160
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0054806, upper bound: 0.0049309
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0054320, upper bound: 0.0049379
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0052959, upper bound: 0.0050429
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0052838, upper bound: 0.0050688
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0051126, upper bound: 0.0051993
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0050677, upper bound: 0.0052068
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0048783, upper bound: 0.0052613
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0048673, upper bound: 0.0052848
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0052652, upper bound: 0.0050120
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0052484, upper bound: 0.0050200
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0051719, upper bound: 0.0052024
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0051677, upper bound: 0.0052407
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0049581, upper bound: 0.0053239
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0049372, upper bound: 0.0053432
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0048159, upper bound: 0.0054491
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0048052, upper bound: 0.0055161
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0054852, upper bound: 0.0049323
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0054386, upper bound: 0.0049388
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0053010, upper bound: 0.0050464
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0052862, upper bound: 0.0050701
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0051193, upper bound: 0.0052012
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0050734, upper bound: 0.0052080
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0048838, upper bound: 0.0052647
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0048735, upper bound: 0.0052864
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0052698, upper bound: 0.0050119
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0052524, upper bound: 0.0050209
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0051751, upper bound: 0.0052037
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0051714, upper bound: 0.0052445
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0049646, upper bound: 0.0053239
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0049472, upper bound: 0.0053432
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0048195, upper bound: 0.0054519
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0048117, upper bound: 0.0055170
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0053850, upper bound: 0.0049483
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0053382, upper bound: 0.0049511
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0052328, upper bound: 0.0050979
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0052185, upper bound: 0.0051178
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0050766, upper bound: 0.0052829
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.53
Output dim: 4, lower bound: -0.0050302, upper bound: 0.0052893
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0049001, upper bound: 0.0054158
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0052020, upper bound: 0.0050542
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0051286, upper bound: 0.0053329
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0049563, upper bound: 0.0054543
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0048308, upper bound: 0.0056633
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0054149, upper bound: 0.0049767
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0052610, upper bound: 0.0051438
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0051061, upper bound: 0.0053151
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0049053, upper bound: 0.0054160
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0052020, upper bound: 0.0050542
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0051310, upper bound: 0.0053329
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0049581, upper bound: 0.0054546
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0048339, upper bound: 0.0056615
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0056615, upper bound: 0.0048339
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0054546, upper bound: 0.0049581
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0053329, upper bound: 0.0051310
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0050542, upper bound: 0.0052020
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0054160, upper bound: 0.0049053
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0053152, upper bound: 0.0051061
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0051438, upper bound: 0.0052610
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0049767, upper bound: 0.0054149
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0056633, upper bound: 0.0048308
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0054543, upper bound: 0.0049563
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0053329, upper bound: 0.0051286
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0050542, upper bound: 0.0052020
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0054158, upper bound: 0.0049001
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0053151, upper bound: 0.0051052
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0051435, upper bound: 0.0052596
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0049763, upper bound: 0.0054136
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0055429, upper bound: 0.0048471
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0053679, upper bound: 0.0049925
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0052701, upper bound: 0.0052027
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0050449, upper bound: 0.0052985
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0053119, upper bound: 0.0049108
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0052332, upper bound: 0.0051480
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0050955, upper bound: 0.0053284
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0049640, upper bound: 0.0055141
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0055419, upper bound: 0.0048433
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0053679, upper bound: 0.0049862
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0052660, upper bound: 0.0051992
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0050439, upper bound: 0.0052940
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0053104, upper bound: 0.0049053
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0052322, upper bound: 0.0051414
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0050943, upper bound: 0.0053236
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0049630, upper bound: 0.0055097
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0055449, upper bound: 0.0049523
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0053641, upper bound: 0.0050874
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0051841, upper bound: 0.0052210
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0049343, upper bound: 0.0053032
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0053257, upper bound: 0.0050357
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0052471, upper bound: 0.0052617
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0050107, upper bound: 0.0053633
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0048683, upper bound: 0.0055413
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0055547, upper bound: 0.0049539
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0053656, upper bound: 0.0050885
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0051912, upper bound: 0.0052220
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0049376, upper bound: 0.0053047
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0053286, upper bound: 0.0050359
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0052479, upper bound: 0.0052662
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0050227, upper bound: 0.0053635
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0048717, upper bound: 0.0055422
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0054219, upper bound: 0.0049618
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0052750, upper bound: 0.0051282
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0051279, upper bound: 0.0052892
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0049222, upper bound: 0.0053886
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0052112, upper bound: 0.0050418
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0051535, upper bound: 0.0053165
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0049696, upper bound: 0.0054304
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0048465, upper bound: 0.0056479
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0054230, upper bound: 0.0049633
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0052780, upper bound: 0.0051284
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0051284, upper bound: 0.0052892
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0049251, upper bound: 0.0053885
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0052112, upper bound: 0.0050423
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0051541, upper bound: 0.0053169
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0049740, upper bound: 0.0054301
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 4, lower bound: -0.0048525, upper bound: 0.0056478

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 3.19 + 597.52 = 600.72 seconds
