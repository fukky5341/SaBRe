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
Threshold: 0.00365364


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0023647, 0.0023647)
1: (-0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0060008, 0.0060008)
2: (0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0037229, 0.0037229)
3: (0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0069516, 0.0069516)
4: (-0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0061038, 0.0061038)
5: (0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0023119, 0.0023119)
6: (0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0088225, 0.0088225)
7: (0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0061736, 0.0061736)
8: (-0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0066190, 0.0066190)
9: (-0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0043722, 0.0043722)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.55 + 2.31 = 3.87 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0039233, upper bound: 0.0039234

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0039157, upper bound: 0.0039072
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0039072, upper bound: 0.0039157
time: 1.24 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.35 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.35
Output dim: 7, lower bound: -0.0039157, upper bound: 0.0039072
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.35
Output dim: 7, lower bound: -0.0039072, upper bound: 0.0039157

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0023529, 0.0023558
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0059707, 0.0059781
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0037043, 0.0037088
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0069253, 0.0069168
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0060733, 0.0060807
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0023004, 0.0023032
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0087891, 0.0087783
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0061502, 0.0061427
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0065940, 0.0065859
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0043504, 0.0043557

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038930, upper bound: 0.0038620
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038637, upper bound: 0.0038829
time: 1.26 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0023558, 0.0023529
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0059781, 0.0059707
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0037088, 0.0037043
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0069168, 0.0069253
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0060807, 0.0060733
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0023032, 0.0023004
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0087783, 0.0087891
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0061427, 0.0061502
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0065859, 0.0065940
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0043557, 0.0043504

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038829, upper bound: 0.0038637
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038620, upper bound: 0.0038931
time: 1.51 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.19 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.19
Output dim: 7, lower bound: -0.0038930, upper bound: 0.0038620
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.19
Output dim: 7, lower bound: -0.0038637, upper bound: 0.0038829
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.19
Output dim: 7, lower bound: -0.0038829, upper bound: 0.0038637
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.19
Output dim: 7, lower bound: -0.0038620, upper bound: 0.0038931

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0023352, 0.0023489
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0059260, 0.0059607
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036765, 0.0036980
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0069051, 0.0068649
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0060277, 0.0060630
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022831, 0.0022965
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0087635, 0.0087125
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0061323, 0.0060966
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0065748, 0.0065365
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0043177, 0.0043430

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038686, upper bound: 0.0038229
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038549, upper bound: 0.0038373
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0023456, 0.0023381
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0059523, 0.0059333
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036928, 0.0036810
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0068734, 0.0068955
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0060545, 0.0060351
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022933, 0.0022859
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0087233, 0.0087513
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0061041, 0.0061237
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0065446, 0.0065656
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0043369, 0.0043231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038390, upper bound: 0.0038424
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038262, upper bound: 0.0038574
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0023381, 0.0023456
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0059333, 0.0059523
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036810, 0.0036928
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0068955, 0.0068734
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0060351, 0.0060545
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022859, 0.0022933
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0087513, 0.0087233
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0061237, 0.0061041
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0065656, 0.0065446
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0043231, 0.0043369

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038573, upper bound: 0.0038262
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038424, upper bound: 0.0038389
time: 1.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0023489, 0.0023352
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0059607, 0.0059260
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036980, 0.0036765
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0068649, 0.0069051
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0060630, 0.0060277
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022965, 0.0022831
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0087125, 0.0087635
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0060966, 0.0061323
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0065365, 0.0065748
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0043430, 0.0043177

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038372, upper bound: 0.0038549
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038230, upper bound: 0.0038686
time: 1.48 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.70 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.70
Output dim: 7, lower bound: -0.0038686, upper bound: 0.0038229
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.70
Output dim: 7, lower bound: -0.0038549, upper bound: 0.0038373
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.70
Output dim: 7, lower bound: -0.0038390, upper bound: 0.0038424
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.70
Output dim: 7, lower bound: -0.0038262, upper bound: 0.0038574
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.70
Output dim: 7, lower bound: -0.0038573, upper bound: 0.0038262
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.70
Output dim: 7, lower bound: -0.0038424, upper bound: 0.0038389
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.70
Output dim: 7, lower bound: -0.0038372, upper bound: 0.0038549
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.70
Output dim: 7, lower bound: -0.0038230, upper bound: 0.0038686

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0023212, 0.0023377
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0058904, 0.0059321
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036544, 0.0036803
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0068721, 0.0068237
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0059915, 0.0060340
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022694, 0.0022855
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0087216, 0.0086602
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0061029, 0.0060600
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0065433, 0.0064973
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042918, 0.0043222

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038562, upper bound: 0.0038078
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038451, upper bound: 0.0038107
time: 1.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0023240, 0.0023347
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0058974, 0.0059246
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036588, 0.0036756
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0068633, 0.0068319
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0059987, 0.0060263
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022721, 0.0022826
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0087104, 0.0086705
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0060951, 0.0060672
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0065350, 0.0065050
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042969, 0.0043167

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038427, upper bound: 0.0038173
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038349, upper bound: 0.0038247
time: 1.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0023316, 0.0023269
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0059168, 0.0059047
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036708, 0.0036633
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0068404, 0.0068544
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0060184, 0.0060061
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022796, 0.0022750
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0086813, 0.0086991
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0060748, 0.0060872
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0065131, 0.0065264
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0043111, 0.0043023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038263, upper bound: 0.0038223
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038211, upper bound: 0.0038302
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0023344, 0.0023241
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0059238, 0.0058976
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036751, 0.0036589
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0068321, 0.0068624
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0060255, 0.0059989
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022823, 0.0022722
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0086709, 0.0087093
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0060674, 0.0060943
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0065053, 0.0065341
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0043161, 0.0042971

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038140, upper bound: 0.0038340
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038107, upper bound: 0.0038451
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0023241, 0.0023344
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0058976, 0.0059238
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036589, 0.0036751
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0068624, 0.0068321
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0059989, 0.0060255
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022722, 0.0022823
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0087093, 0.0086709
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0060943, 0.0060674
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0065341, 0.0065053
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042971, 0.0043161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038452, upper bound: 0.0038107
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038340, upper bound: 0.0038140
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0023269, 0.0023316
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0059047, 0.0059168
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036633, 0.0036708
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0068544, 0.0068404
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0060061, 0.0060184
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022750, 0.0022796
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0086991, 0.0086813
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0060872, 0.0060748
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0065264, 0.0065131
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0043023, 0.0043111

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038302, upper bound: 0.0038211
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038223, upper bound: 0.0038263
time: 1.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0023347, 0.0023240
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0059246, 0.0058974
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036756, 0.0036588
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0068319, 0.0068633
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0060263, 0.0059987
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022826, 0.0022721
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0086705, 0.0087104
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0060672, 0.0060951
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0065050, 0.0065350
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0043167, 0.0042969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038247, upper bound: 0.0038349
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038173, upper bound: 0.0038427
time: 1.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0023377, 0.0023212
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0059321, 0.0058904
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036803, 0.0036544
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0068237, 0.0068721
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0060340, 0.0059915
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022855, 0.0022694
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0086602, 0.0087216
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0060600, 0.0061029
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0064973, 0.0065433
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0043222, 0.0042918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038107, upper bound: 0.0038451
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038078, upper bound: 0.0038563
time: 1.40 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.41 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.41
Output dim: 7, lower bound: -0.0038562, upper bound: 0.0038078
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.41
Output dim: 7, lower bound: -0.0038451, upper bound: 0.0038107
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.41
Output dim: 7, lower bound: -0.0038427, upper bound: 0.0038173
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.41
Output dim: 7, lower bound: -0.0038349, upper bound: 0.0038247
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.41
Output dim: 7, lower bound: -0.0038263, upper bound: 0.0038223
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.41
Output dim: 7, lower bound: -0.0038211, upper bound: 0.0038302
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.41
Output dim: 7, lower bound: -0.0038140, upper bound: 0.0038340
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.41
Output dim: 7, lower bound: -0.0038107, upper bound: 0.0038451
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.41
Output dim: 7, lower bound: -0.0038452, upper bound: 0.0038107
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.41
Output dim: 7, lower bound: -0.0038340, upper bound: 0.0038140
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.41
Output dim: 7, lower bound: -0.0038302, upper bound: 0.0038211
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.41
Output dim: 7, lower bound: -0.0038223, upper bound: 0.0038263
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.41
Output dim: 7, lower bound: -0.0038247, upper bound: 0.0038349
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.41
Output dim: 7, lower bound: -0.0038173, upper bound: 0.0038427
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.41
Output dim: 7, lower bound: -0.0038107, upper bound: 0.0038451
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.41
Output dim: 7, lower bound: -0.0038078, upper bound: 0.0038563

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0023176, 0.0023369
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0058811, 0.0059301
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036487, 0.0036791
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0068697, 0.0068130
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0059821, 0.0060319
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022659, 0.0022847
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0087186, 0.0086466
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0061009, 0.0060505
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0065411, 0.0064871
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042851, 0.0043208

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038357, upper bound: 0.0037481
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037796, upper bound: 0.0037871
time: 1.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0023204, 0.0023336
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0058884, 0.0059217
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036532, 0.0036739
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0068600, 0.0068214
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0059895, 0.0060234
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022686, 0.0022815
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0087063, 0.0086572
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0060922, 0.0060579
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0065318, 0.0064950
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042903, 0.0043146

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038242, upper bound: 0.0037482
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037794, upper bound: 0.0037909
time: 1.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0023206, 0.0023339
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0058888, 0.0059225
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036535, 0.0036744
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0068610, 0.0068219
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0059899, 0.0060242
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022688, 0.0022818
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0087075, 0.0086579
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0060931, 0.0060584
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0065327, 0.0064955
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042907, 0.0043152

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037741, upper bound: 0.0037540
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037741, upper bound: 0.0037957
time: 1.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0023232, 0.0023306
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0058954, 0.0059142
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036575, 0.0036692
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0068513, 0.0068296
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0059966, 0.0060158
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022714, 0.0022786
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0086952, 0.0086676
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0060845, 0.0060652
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0065235, 0.0065028
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042955, 0.0043092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038148, upper bound: 0.0037539
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037729, upper bound: 0.0038038
time: 1.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0023278, 0.0023261
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0059071, 0.0059027
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036648, 0.0036621
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0068380, 0.0068431
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0060086, 0.0060041
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022759, 0.0022742
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0086783, 0.0086848
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0060727, 0.0060772
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0065109, 0.0065157
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0043040, 0.0043008

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037550, upper bound: 0.0037647
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037550, upper bound: 0.0038013
time: 1.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0023308, 0.0023232
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0059148, 0.0058955
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036696, 0.0036576
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0068297, 0.0068520
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0060163, 0.0059967
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022788, 0.0022714
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0086677, 0.0086961
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0060652, 0.0060851
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0065029, 0.0065242
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0043096, 0.0042955

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037994, upper bound: 0.0037662
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037552, upper bound: 0.0038096
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0023307, 0.0023233
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0059146, 0.0058956
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036694, 0.0036577
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0068298, 0.0068517
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0060161, 0.0059968
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022787, 0.0022714
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0086679, 0.0086957
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0060654, 0.0060849
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0065030, 0.0065239
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0043094, 0.0042956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037494, upper bound: 0.0037720
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037494, upper bound: 0.0038124
time: 1.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0023336, 0.0023201
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0059218, 0.0058876
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036739, 0.0036527
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0068205, 0.0068601
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0060234, 0.0059887
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022815, 0.0022683
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0086561, 0.0087063
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0060571, 0.0060923
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0064942, 0.0065319
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0043147, 0.0042898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037495, upper bound: 0.0037721
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037495, upper bound: 0.0038245
time: 1.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0023201, 0.0023336
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0058876, 0.0059218
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036527, 0.0036739
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0068601, 0.0068205
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0059887, 0.0060234
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022683, 0.0022815
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0087063, 0.0086561
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0060923, 0.0060571
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0065319, 0.0064942
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042898, 0.0043147

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038244, upper bound: 0.0037495
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037721, upper bound: 0.0037908
time: 1.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0023233, 0.0023307
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0058956, 0.0059146
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036577, 0.0036694
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0068517, 0.0068298
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0059968, 0.0060161
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022714, 0.0022787
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0086957, 0.0086679
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0060849, 0.0060654
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0065239, 0.0065030
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042956, 0.0043094

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038125, upper bound: 0.0037494
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037721, upper bound: 0.0037944
time: 1.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0023232, 0.0023308
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0058955, 0.0059148
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036576, 0.0036696
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0068520, 0.0068297
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0059967, 0.0060163
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022714, 0.0022788
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0086961, 0.0086677
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0060851, 0.0060652
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0065242, 0.0065029
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042955, 0.0043096

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038096, upper bound: 0.0037551
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037662, upper bound: 0.0037994
time: 1.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0023261, 0.0023278
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0059027, 0.0059071
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036621, 0.0036648
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0068431, 0.0068380
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0060041, 0.0060086
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022742, 0.0022759
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0086848, 0.0086783
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0060772, 0.0060727
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0065157, 0.0065109
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0043008, 0.0043040

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038013, upper bound: 0.0037550
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037647, upper bound: 0.0038058
time: 1.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0023306, 0.0023232
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0059142, 0.0058954
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036692, 0.0036575
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0068296, 0.0068513
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0060158, 0.0059966
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022786, 0.0022714
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0086676, 0.0086952
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0060652, 0.0060845
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0065028, 0.0065235
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0043092, 0.0042955

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038038, upper bound: 0.0037729
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037539, upper bound: 0.0038148
time: 1.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0023339, 0.0023206
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0059225, 0.0058888
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036744, 0.0036535
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0068219, 0.0068610
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0060242, 0.0059899
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022818, 0.0022688
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0086579, 0.0087075
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0060584, 0.0060931
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0064955, 0.0065327
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0043152, 0.0042907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037956, upper bound: 0.0037742
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037541, upper bound: 0.0038228
time: 1.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0023336, 0.0023204
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0059217, 0.0058884
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036739, 0.0036532
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0068214, 0.0068600
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0060234, 0.0059895
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022815, 0.0022686
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0086572, 0.0087063
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0060579, 0.0060922
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0064950, 0.0065318
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0043146, 0.0042903

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037909, upper bound: 0.0037795
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037482, upper bound: 0.0038242
time: 1.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0023369, 0.0023176
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0059301, 0.0058811
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036791, 0.0036487
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0068130, 0.0068697
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0060319, 0.0059821
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022847, 0.0022659
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0086466, 0.0087186
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0060505, 0.0061009
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0064871, 0.0065411
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0043208, 0.0042851

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037482, upper bound: 0.0037796
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037482, upper bound: 0.0038357
time: 1.44 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.56 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 7, lower bound: -0.0038357, upper bound: 0.0037481
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 7, lower bound: -0.0037796, upper bound: 0.0037871
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 7, lower bound: -0.0038242, upper bound: 0.0037482
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 7, lower bound: -0.0037794, upper bound: 0.0037909
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 7, lower bound: -0.0037741, upper bound: 0.0037540
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 7, lower bound: -0.0037741, upper bound: 0.0037957
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 7, lower bound: -0.0038148, upper bound: 0.0037539
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 7, lower bound: -0.0037729, upper bound: 0.0038038
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 7, lower bound: -0.0037550, upper bound: 0.0037647
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 7, lower bound: -0.0037550, upper bound: 0.0038013
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 7, lower bound: -0.0037994, upper bound: 0.0037662
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 7, lower bound: -0.0037552, upper bound: 0.0038096
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 7, lower bound: -0.0037494, upper bound: 0.0037720
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 7, lower bound: -0.0037494, upper bound: 0.0038124
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 7, lower bound: -0.0037495, upper bound: 0.0037721
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 7, lower bound: -0.0037495, upper bound: 0.0038245
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 7, lower bound: -0.0038244, upper bound: 0.0037495
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 7, lower bound: -0.0037721, upper bound: 0.0037908
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 7, lower bound: -0.0038125, upper bound: 0.0037494
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 7, lower bound: -0.0037721, upper bound: 0.0037944
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 7, lower bound: -0.0038096, upper bound: 0.0037551
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 7, lower bound: -0.0037662, upper bound: 0.0037994
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 7, lower bound: -0.0038013, upper bound: 0.0037550
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 7, lower bound: -0.0037647, upper bound: 0.0038058
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 7, lower bound: -0.0038038, upper bound: 0.0037729
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 7, lower bound: -0.0037539, upper bound: 0.0038148
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 7, lower bound: -0.0037956, upper bound: 0.0037742
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 7, lower bound: -0.0037541, upper bound: 0.0038228
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 7, lower bound: -0.0037909, upper bound: 0.0037795
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 7, lower bound: -0.0037482, upper bound: 0.0038242
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 7, lower bound: -0.0037482, upper bound: 0.0037796
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 7, lower bound: -0.0037482, upper bound: 0.0038357

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022627, 0.0022930
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057418, 0.0058187
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035622, 0.0036099
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0067407, 0.0066516
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058404, 0.0059186
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022122, 0.0022418
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0085548, 0.0084418
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059862, 0.0059071
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0064182, 0.0063334
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0041836, 0.0042396

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037498, upper bound: 0.0036614
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037396, upper bound: 0.0036693
time: 1.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022730, 0.0022820
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057681, 0.0057908
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035786, 0.0035926
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0067084, 0.0066821
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058671, 0.0058902
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022223, 0.0022311
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0085138, 0.0084804
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059575, 0.0059342
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063874, 0.0063624
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042027, 0.0042192

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036980, upper bound: 0.0036951
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036908, upper bound: 0.0037020
time: 1.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022655, 0.0022895
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057491, 0.0058100
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035667, 0.0036045
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0067306, 0.0066600
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058478, 0.0059097
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022150, 0.0022384
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0085420, 0.0084524
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059773, 0.0059146
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0064086, 0.0063414
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0041888, 0.0042332

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037399, upper bound: 0.0036616
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037324, upper bound: 0.0036694
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022759, 0.0022787
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057754, 0.0057824
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035831, 0.0035874
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066987, 0.0066905
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058746, 0.0058817
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022251, 0.0022278
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0085015, 0.0084912
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059489, 0.0059417
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063782, 0.0063704
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042080, 0.0042131

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036978, upper bound: 0.0037001
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036904, upper bound: 0.0037073
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022657, 0.0022897
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057495, 0.0058105
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035670, 0.0036049
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0067312, 0.0066605
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058482, 0.0059103
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022151, 0.0022387
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0085428, 0.0084531
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059778, 0.0059151
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0064092, 0.0063419
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0041892, 0.0042336

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037355, upper bound: 0.0036657
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037310, upper bound: 0.0036762
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022762, 0.0022790
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057761, 0.0057832
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035835, 0.0035879
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066996, 0.0066913
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058752, 0.0058825
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022254, 0.0022281
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0085027, 0.0084921
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059497, 0.0059424
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063791, 0.0063712
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042085, 0.0042137

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036914, upper bound: 0.0037021
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036855, upper bound: 0.0037118
time: 1.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022683, 0.0022863
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057561, 0.0058017
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035711, 0.0035994
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0067210, 0.0066682
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058549, 0.0059013
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022177, 0.0022353
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0085299, 0.0084628
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059688, 0.0059218
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063995, 0.0063491
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0041940, 0.0042272

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037283, upper bound: 0.0036660
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037226, upper bound: 0.0036763
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022791, 0.0022757
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057835, 0.0057749
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035881, 0.0035828
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066899, 0.0066999
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058827, 0.0058740
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022282, 0.0022249
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084904, 0.0085030
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059412, 0.0059500
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063699, 0.0063793
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042139, 0.0042077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036908, upper bound: 0.0037077
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036847, upper bound: 0.0037202
time: 1.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022729, 0.0022822
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057678, 0.0057915
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035784, 0.0035931
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0067092, 0.0066818
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058669, 0.0058910
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022222, 0.0022313
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0085148, 0.0084800
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059583, 0.0059339
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063882, 0.0063621
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042025, 0.0042198

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037229, upper bound: 0.0036794
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037092, upper bound: 0.0036850
time: 1.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022829, 0.0022712
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057933, 0.0057634
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035942, 0.0035756
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066766, 0.0067112
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058927, 0.0058624
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022320, 0.0022205
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084735, 0.0085174
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059294, 0.0059601
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063572, 0.0063902
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042211, 0.0041993

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036766, upper bound: 0.0037129
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036663, upper bound: 0.0037173
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022759, 0.0022793
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057755, 0.0057840
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035831, 0.0035884
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0067005, 0.0066906
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058746, 0.0058833
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022252, 0.0022284
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0085038, 0.0084913
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059506, 0.0059418
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063799, 0.0063705
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042081, 0.0042143

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037160, upper bound: 0.0036811
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037053, upper bound: 0.0036868
time: 1.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022862, 0.0022683
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0058017, 0.0057562
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035994, 0.0035712
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066683, 0.0067210
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0059013, 0.0058550
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022352, 0.0022177
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084629, 0.0085298
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059219, 0.0059687
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063492, 0.0063994
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042272, 0.0041940

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036765, upper bound: 0.0037221
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036661, upper bound: 0.0037246
time: 1.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022758, 0.0022791
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057753, 0.0057837
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035830, 0.0035882
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0067001, 0.0066904
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058744, 0.0058830
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022251, 0.0022283
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0085033, 0.0084909
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059502, 0.0059415
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063795, 0.0063703
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042079, 0.0042140

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037106, upper bound: 0.0036846
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037026, upper bound: 0.0036931
time: 1.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022859, 0.0022684
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0058009, 0.0057563
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035989, 0.0035712
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066684, 0.0067200
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0059005, 0.0058551
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022349, 0.0022178
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084631, 0.0085286
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059220, 0.0059679
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063494, 0.0063985
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042266, 0.0041941

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036699, upper bound: 0.0037238
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036619, upper bound: 0.0037292
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022787, 0.0022761
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057825, 0.0057759
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035875, 0.0035834
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066911, 0.0066987
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058817, 0.0058751
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022278, 0.0022253
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084919, 0.0085015
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059422, 0.0059490
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063710, 0.0063782
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042132, 0.0042084

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037059, upper bound: 0.0036857
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036990, upper bound: 0.0036934
time: 1.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022894, 0.0022652
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0058096, 0.0057483
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036043, 0.0035662
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066591, 0.0067302
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0059093, 0.0058470
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022383, 0.0022147
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084512, 0.0085414
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059138, 0.0059769
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063405, 0.0064082
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042330, 0.0041883

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036698, upper bound: 0.0037327
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036618, upper bound: 0.0037396
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022652, 0.0022894
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057483, 0.0058096
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035662, 0.0036043
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0067302, 0.0066591
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058470, 0.0059093
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022147, 0.0022383
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0085414, 0.0084512
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059769, 0.0059138
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0064082, 0.0063405
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0041883, 0.0042330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037397, upper bound: 0.0036618
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037328, upper bound: 0.0036698
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022761, 0.0022787
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057759, 0.0057825
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035834, 0.0035875
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066987, 0.0066911
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058751, 0.0058817
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022253, 0.0022278
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0085015, 0.0084919
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059490, 0.0059422
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063782, 0.0063710
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042084, 0.0042132

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036934, upper bound: 0.0036989
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036857, upper bound: 0.0037059
time: 1.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022684, 0.0022859
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057563, 0.0058009
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035712, 0.0035989
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0067200, 0.0066684
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058551, 0.0059005
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022178, 0.0022349
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0085286, 0.0084631
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059679, 0.0059220
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063985, 0.0063494
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0041941, 0.0042266

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037293, upper bound: 0.0036619
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037239, upper bound: 0.0036699
time: 1.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022791, 0.0022758
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057837, 0.0057753
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035882, 0.0035830
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066904, 0.0067001
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058830, 0.0058744
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022283, 0.0022251
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084909, 0.0085033
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059415, 0.0059502
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063703, 0.0063795
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042140, 0.0042079

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036931, upper bound: 0.0037025
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036846, upper bound: 0.0037106
time: 1.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022683, 0.0022862
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057562, 0.0058017
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035712, 0.0035994
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0067210, 0.0066683
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058550, 0.0059013
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022177, 0.0022352
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0085298, 0.0084629
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059687, 0.0059219
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063994, 0.0063492
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0041940, 0.0042272

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037247, upper bound: 0.0036661
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037221, upper bound: 0.0036765
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022793, 0.0022759
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057840, 0.0057755
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035884, 0.0035831
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066906, 0.0067005
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058833, 0.0058746
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022284, 0.0022252
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084913, 0.0085038
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059418, 0.0059506
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063705, 0.0063799
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042143, 0.0042081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036868, upper bound: 0.0037053
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036811, upper bound: 0.0037159
time: 1.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022712, 0.0022829
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057634, 0.0057933
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035756, 0.0035942
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0067112, 0.0066766
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058624, 0.0058927
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022205, 0.0022320
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0085174, 0.0084735
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059601, 0.0059294
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063901, 0.0063572
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0041993, 0.0042211

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037173, upper bound: 0.0036663
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037129, upper bound: 0.0036766
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022822, 0.0022729
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057915, 0.0057678
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035931, 0.0035784
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066818, 0.0067092
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058910, 0.0058669
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022313, 0.0022222
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084800, 0.0085148
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059339, 0.0059583
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063621, 0.0063882
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042198, 0.0042025

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036850, upper bound: 0.0037092
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036794, upper bound: 0.0037229
time: 1.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022757, 0.0022791
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057749, 0.0057835
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035828, 0.0035881
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066999, 0.0066899
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058740, 0.0058827
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022249, 0.0022282
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0085030, 0.0084904
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059500, 0.0059412
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063793, 0.0063699
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042077, 0.0042139

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037203, upper bound: 0.0036847
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037077, upper bound: 0.0036908
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022863, 0.0022683
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0058017, 0.0057561
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035994, 0.0035711
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066682, 0.0067210
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0059013, 0.0058549
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022353, 0.0022177
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084628, 0.0085299
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059218, 0.0059688
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063491, 0.0063995
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042272, 0.0041940

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036763, upper bound: 0.0037226
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036661, upper bound: 0.0037283
time: 1.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022790, 0.0022762
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057832, 0.0057761
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035879, 0.0035835
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066913, 0.0066996
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058825, 0.0058752
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022281, 0.0022254
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084921, 0.0085027
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059424, 0.0059497
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063712, 0.0063791
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042137, 0.0042085

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037118, upper bound: 0.0036855
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037021, upper bound: 0.0036914
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022897, 0.0022657
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0058105, 0.0057495
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036049, 0.0035670
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066605, 0.0067312
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0059103, 0.0058482
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022387, 0.0022151
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084531, 0.0085428
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059151, 0.0059778
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063419, 0.0064092
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042336, 0.0041892

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036762, upper bound: 0.0037310
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036657, upper bound: 0.0037355
time: 1.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022787, 0.0022759
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057824, 0.0057754
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035874, 0.0035831
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066905, 0.0066987
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058817, 0.0058746
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022278, 0.0022251
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084912, 0.0085015
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059417, 0.0059489
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063704, 0.0063782
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042131, 0.0042080

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037073, upper bound: 0.0036903
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037001, upper bound: 0.0036978
time: 1.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022895, 0.0022655
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0058100, 0.0057491
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036045, 0.0035667
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066600, 0.0067306
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0059097, 0.0058478
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022384, 0.0022150
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084524, 0.0085420
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059146, 0.0059773
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063414, 0.0064086
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042332, 0.0041888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036694, upper bound: 0.0037324
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036616, upper bound: 0.0037399
time: 1.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022820, 0.0022730
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057908, 0.0057681
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035926, 0.0035786
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066821, 0.0067084
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058902, 0.0058671
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022311, 0.0022223
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084804, 0.0085138
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059342, 0.0059575
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063624, 0.0063874
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042192, 0.0042027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037020, upper bound: 0.0036908
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036951, upper bound: 0.0036980
time: 1.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022930, 0.0022627
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0058187, 0.0057418
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036099, 0.0035622
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066516, 0.0067407
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0059186, 0.0058404
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022418, 0.0022122
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084418, 0.0085548
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059071, 0.0059862
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063334, 0.0064182
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042396, 0.0041836

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036693, upper bound: 0.0037395
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036614, upper bound: 0.0037498
time: 1.54 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.62 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0037498, upper bound: 0.0036614
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0037396, upper bound: 0.0036693
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0036980, upper bound: 0.0036951
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0036908, upper bound: 0.0037020
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0037399, upper bound: 0.0036616
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0037324, upper bound: 0.0036694
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0036978, upper bound: 0.0037001
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0036904, upper bound: 0.0037073
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0037355, upper bound: 0.0036657
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0037310, upper bound: 0.0036762
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0036914, upper bound: 0.0037021
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0036855, upper bound: 0.0037118
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0037283, upper bound: 0.0036660
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0037226, upper bound: 0.0036763
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0036908, upper bound: 0.0037077
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0036847, upper bound: 0.0037202
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0037229, upper bound: 0.0036794
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0037092, upper bound: 0.0036850
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0036766, upper bound: 0.0037129
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0036663, upper bound: 0.0037173
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0037160, upper bound: 0.0036811
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0037053, upper bound: 0.0036868
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0036765, upper bound: 0.0037221
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0036661, upper bound: 0.0037246
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0037106, upper bound: 0.0036846
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0037026, upper bound: 0.0036931
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0036699, upper bound: 0.0037238
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0036619, upper bound: 0.0037292
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0037059, upper bound: 0.0036857
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0036990, upper bound: 0.0036934
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0036698, upper bound: 0.0037327
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0036618, upper bound: 0.0037396
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0037397, upper bound: 0.0036618
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0037328, upper bound: 0.0036698
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0036934, upper bound: 0.0036989
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0036857, upper bound: 0.0037059
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0037293, upper bound: 0.0036619
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0037239, upper bound: 0.0036699
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0036931, upper bound: 0.0037025
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0036846, upper bound: 0.0037106
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0037247, upper bound: 0.0036661
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0037221, upper bound: 0.0036765
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0036868, upper bound: 0.0037053
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0036811, upper bound: 0.0037159
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0037173, upper bound: 0.0036663
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0037129, upper bound: 0.0036766
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0036850, upper bound: 0.0037092
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0036794, upper bound: 0.0037229
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0037203, upper bound: 0.0036847
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0037077, upper bound: 0.0036908
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0036763, upper bound: 0.0037226
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0036661, upper bound: 0.0037283
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0037118, upper bound: 0.0036855
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0037021, upper bound: 0.0036914
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0036762, upper bound: 0.0037310
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0036657, upper bound: 0.0037355
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0037073, upper bound: 0.0036903
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0037001, upper bound: 0.0036978
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0036694, upper bound: 0.0037324
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0036616, upper bound: 0.0037399
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0037020, upper bound: 0.0036908
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0036951, upper bound: 0.0036980
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0036693, upper bound: 0.0037395
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 7, lower bound: -0.0036614, upper bound: 0.0037498

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022582, 0.0022894
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057305, 0.0058098
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035552, 0.0036044
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0067304, 0.0066385
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058289, 0.0059095
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022078, 0.0022384
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0085417, 0.0084251
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059771, 0.0058955
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0064084, 0.0063209
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0041753, 0.0042331

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036540, upper bound: 0.0035735
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036539, upper bound: 0.0035695
time: 1.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022627, 0.0022885
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057418, 0.0058074
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035622, 0.0036029
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0067276, 0.0066516
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058404, 0.0059071
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022122, 0.0022374
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0085382, 0.0084418
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059746, 0.0059071
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0064057, 0.0063334
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0041836, 0.0042313

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036463, upper bound: 0.0035799
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036463, upper bound: 0.0035768
time: 1.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022686, 0.0022784
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057568, 0.0057818
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035715, 0.0035870
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066979, 0.0066689
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058556, 0.0058811
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022179, 0.0022276
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0085005, 0.0084638
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059483, 0.0059225
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063775, 0.0063499
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0041945, 0.0042127

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036058, upper bound: 0.0036083
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036067, upper bound: 0.0036021
time: 1.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022730, 0.0022775
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057681, 0.0057795
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035786, 0.0035856
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066952, 0.0066821
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058671, 0.0058787
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022223, 0.0022267
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084971, 0.0084804
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059459, 0.0059342
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063749, 0.0063624
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042027, 0.0042110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035989, upper bound: 0.0036147
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036001, upper bound: 0.0036091
time: 1.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022610, 0.0022858
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057377, 0.0058006
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035597, 0.0035987
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0067198, 0.0066469
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058362, 0.0059002
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022106, 0.0022348
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0085282, 0.0084358
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059677, 0.0059029
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063983, 0.0063289
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0041806, 0.0042264

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036470, upper bound: 0.0035735
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036478, upper bound: 0.0035697
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022655, 0.0022851
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057491, 0.0057987
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035667, 0.0035975
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0067175, 0.0066600
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058478, 0.0058982
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022150, 0.0022341
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0085253, 0.0084524
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059656, 0.0059146
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063961, 0.0063414
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0041888, 0.0042250

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036397, upper bound: 0.0035798
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036408, upper bound: 0.0035768
time: 1.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022714, 0.0022750
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057641, 0.0057730
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035761, 0.0035816
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066878, 0.0066774
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058630, 0.0058721
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022208, 0.0022242
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084876, 0.0084745
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059392, 0.0059301
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063678, 0.0063580
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0041998, 0.0042063

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036060, upper bound: 0.0036113
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036068, upper bound: 0.0036074
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022759, 0.0022742
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057754, 0.0057711
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035831, 0.0035804
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066855, 0.0066905
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058746, 0.0058702
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022251, 0.0022235
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084848, 0.0084912
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059373, 0.0059417
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063657, 0.0063704
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042080, 0.0042049

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035989, upper bound: 0.0036178
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036001, upper bound: 0.0036146
time: 1.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022612, 0.0022862
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057382, 0.0058016
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035600, 0.0035994
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0067209, 0.0066474
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058367, 0.0059012
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022108, 0.0022352
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0085297, 0.0084364
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059687, 0.0059034
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063994, 0.0063294
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0041809, 0.0042271

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036435, upper bound: 0.0035745
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036463, upper bound: 0.0035727
time: 1.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022657, 0.0022853
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057495, 0.0057992
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035670, 0.0035978
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0067181, 0.0066605
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058482, 0.0058988
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022151, 0.0022343
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0085261, 0.0084531
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059662, 0.0059151
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063967, 0.0063419
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0041892, 0.0042254

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036393, upper bound: 0.0035824
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036422, upper bound: 0.0035812
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022717, 0.0022754
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057648, 0.0057742
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035765, 0.0035823
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066891, 0.0066782
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058637, 0.0058733
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022210, 0.0022247
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084894, 0.0084755
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059405, 0.0059307
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063691, 0.0063587
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042003, 0.0042072

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036005, upper bound: 0.0036105
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036040, upper bound: 0.0036092
time: 1.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022762, 0.0022745
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057761, 0.0057719
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035835, 0.0035809
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066865, 0.0066913
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058752, 0.0058710
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022254, 0.0022238
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084860, 0.0084921
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059381, 0.0059424
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063666, 0.0063712
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042085, 0.0042055

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035948, upper bound: 0.0036193
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035979, upper bound: 0.0036182
time: 1.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022638, 0.0022825
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057448, 0.0057922
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035641, 0.0035935
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0067100, 0.0066550
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058434, 0.0058916
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022133, 0.0022316
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0085158, 0.0084461
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059590, 0.0059102
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063889, 0.0063366
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0041857, 0.0042203

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036365, upper bound: 0.0035746
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036406, upper bound: 0.0035727
time: 1.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022683, 0.0022818
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057561, 0.0057904
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035711, 0.0035924
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0067079, 0.0066682
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058549, 0.0058898
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022177, 0.0022309
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0085132, 0.0084628
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059571, 0.0059218
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063870, 0.0063491
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0041940, 0.0042190

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036310, upper bound: 0.0035823
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036360, upper bound: 0.0035811
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022746, 0.0022719
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057721, 0.0057653
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035811, 0.0035768
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066788, 0.0066867
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058712, 0.0058643
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022239, 0.0022212
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084763, 0.0084863
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059313, 0.0059383
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063593, 0.0063668
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042056, 0.0042007

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036001, upper bound: 0.0036146
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036039, upper bound: 0.0036139
time: 1.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022791, 0.0022712
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057835, 0.0057636
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035881, 0.0035757
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066768, 0.0066999
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058827, 0.0058625
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022282, 0.0022206
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084738, 0.0085030
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059295, 0.0059500
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063574, 0.0063793
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042139, 0.0041994

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035938, upper bound: 0.0036238
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035978, upper bound: 0.0036238
time: 1.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022684, 0.0022786
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057565, 0.0057823
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035714, 0.0035874
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066986, 0.0066686
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058553, 0.0058816
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022178, 0.0022278
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0085014, 0.0084634
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059488, 0.0059222
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063781, 0.0063496
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0041943, 0.0042131

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036271, upper bound: 0.0035931
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036271, upper bound: 0.0035883
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022729, 0.0022778
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057678, 0.0057802
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035784, 0.0035861
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066961, 0.0066818
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058669, 0.0058794
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022222, 0.0022270
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084982, 0.0084800
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059466, 0.0059339
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063757, 0.0063621
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042025, 0.0042115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036153, upper bound: 0.0035980
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036165, upper bound: 0.0035936
time: 1.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022785, 0.0022677
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057820, 0.0057546
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035871, 0.0035702
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066664, 0.0066981
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058812, 0.0058534
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022276, 0.0022171
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084605, 0.0085008
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059203, 0.0059484
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063475, 0.0063777
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042128, 0.0041929

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035816, upper bound: 0.0036251
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035829, upper bound: 0.0036205
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022829, 0.0022667
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057933, 0.0057521
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035942, 0.0035686
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066635, 0.0067112
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058927, 0.0058508
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022320, 0.0022161
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084569, 0.0085174
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059177, 0.0059601
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063447, 0.0063902
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042211, 0.0041910

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035731, upper bound: 0.0036302
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035746, upper bound: 0.0036249
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022715, 0.0022758
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057642, 0.0057751
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035761, 0.0035829
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066902, 0.0066775
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058631, 0.0058742
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022208, 0.0022250
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084907, 0.0084746
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059414, 0.0059301
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063701, 0.0063580
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0041998, 0.0042078

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036228, upper bound: 0.0035934
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036245, upper bound: 0.0035898
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022759, 0.0022748
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057755, 0.0057727
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035831, 0.0035814
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066874, 0.0066906
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058746, 0.0058718
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022252, 0.0022241
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084872, 0.0084913
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059389, 0.0059418
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063675, 0.0063705
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042081, 0.0042061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036126, upper bound: 0.0035981
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036146, upper bound: 0.0035949
time: 1.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022818, 0.0022650
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057903, 0.0057477
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035924, 0.0035659
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066584, 0.0067078
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058897, 0.0058463
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022309, 0.0022144
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084504, 0.0085131
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059131, 0.0059571
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063398, 0.0063869
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042189, 0.0041878

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035818, upper bound: 0.0036336
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035830, upper bound: 0.0036295
time: 1.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022862, 0.0022639
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0058017, 0.0057449
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035994, 0.0035641
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066551, 0.0067210
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0059013, 0.0058435
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022352, 0.0022134
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084462, 0.0085298
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059103, 0.0059687
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063367, 0.0063994
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042272, 0.0041858

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035731, upper bound: 0.0036361
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035746, upper bound: 0.0036317
time: 1.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022714, 0.0022756
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057639, 0.0057748
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035760, 0.0035827
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066898, 0.0066772
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058629, 0.0058739
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022207, 0.0022249
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084902, 0.0084743
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059410, 0.0059299
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063697, 0.0063578
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0041997, 0.0042076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036185, upper bound: 0.0035951
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036213, upper bound: 0.0035935
time: 1.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022758, 0.0022747
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057753, 0.0057723
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035830, 0.0035812
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066870, 0.0066904
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058744, 0.0058714
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022251, 0.0022239
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084866, 0.0084909
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059385, 0.0059415
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063670, 0.0063703
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042079, 0.0042058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036103, upper bound: 0.0036010
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035735, upper bound: 0.0036004
time: 1.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022815, 0.0022650
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057895, 0.0057477
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035919, 0.0035659
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066585, 0.0067069
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058889, 0.0058464
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022306, 0.0022145
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084504, 0.0085119
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059132, 0.0059562
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063399, 0.0063860
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042183, 0.0041879

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035777, upper bound: 0.0036305
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035807, upper bound: 0.0036302
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022859, 0.0022639
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0058009, 0.0057450
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035989, 0.0035642
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066553, 0.0067200
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0059005, 0.0058436
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022349, 0.0022134
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084464, 0.0085286
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059104, 0.0059679
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063369, 0.0063985
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042266, 0.0041859

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035700, upper bound: 0.0036368
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035735, upper bound: 0.0036359
time: 1.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022742, 0.0022726
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057711, 0.0057672
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035804, 0.0035780
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066810, 0.0066856
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058702, 0.0058662
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022235, 0.0022219
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084790, 0.0084849
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059332, 0.0059373
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063613, 0.0063657
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042049, 0.0042020

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036137, upper bound: 0.0035952
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036189, upper bound: 0.0035940
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022787, 0.0022716
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057825, 0.0057646
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035875, 0.0035764
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066780, 0.0066987
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058817, 0.0058636
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022278, 0.0022210
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084753, 0.0085015
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059306, 0.0059490
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063585, 0.0063782
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042132, 0.0042002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036065, upper bound: 0.0036009
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036119, upper bound: 0.0036006
time: 1.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022849, 0.0022620
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057983, 0.0057400
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035973, 0.0035611
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066496, 0.0067170
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058978, 0.0058386
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022339, 0.0022115
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084392, 0.0085248
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059053, 0.0059652
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063314, 0.0063957
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042247, 0.0041823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035776, upper bound: 0.0036390
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035807, upper bound: 0.0036387
time: 1.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022894, 0.0022607
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0058096, 0.0057369
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036043, 0.0035592
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066460, 0.0067302
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0059093, 0.0058354
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022383, 0.0022103
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084346, 0.0085414
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059021, 0.0059769
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063280, 0.0064082
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042330, 0.0041800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035700, upper bound: 0.0036437
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035737, upper bound: 0.0036436
time: 1.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022607, 0.0022857
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057369, 0.0058004
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035592, 0.0035986
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0067195, 0.0066460
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058354, 0.0058999
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022103, 0.0022347
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0085278, 0.0084346
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059674, 0.0059021
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063980, 0.0063280
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0041800, 0.0042262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036436, upper bound: 0.0035737
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036437, upper bound: 0.0035700
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022652, 0.0022849
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057483, 0.0057983
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035662, 0.0035973
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0067170, 0.0066591
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058470, 0.0058978
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022147, 0.0022339
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0085248, 0.0084512
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059652, 0.0059138
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063957, 0.0063405
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0041883, 0.0042247

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036387, upper bound: 0.0035807
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036390, upper bound: 0.0035776
time: 1.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022716, 0.0022751
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057646, 0.0057733
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035764, 0.0035818
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066881, 0.0066780
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058636, 0.0058725
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022210, 0.0022243
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084881, 0.0084753
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059396, 0.0059306
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063682, 0.0063585
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042002, 0.0042065

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036004, upper bound: 0.0036119
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036009, upper bound: 0.0036065
time: 1.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022761, 0.0022742
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057759, 0.0057711
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035834, 0.0035804
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066856, 0.0066911
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058751, 0.0058702
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022253, 0.0022235
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084849, 0.0084919
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059373, 0.0059422
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063657, 0.0063710
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042084, 0.0042049

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035940, upper bound: 0.0036188
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035952, upper bound: 0.0036137
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022639, 0.0022821
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057450, 0.0057912
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035642, 0.0035929
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0067089, 0.0066553
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058436, 0.0058907
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022134, 0.0022312
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0085144, 0.0084464
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059580, 0.0059104
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063879, 0.0063369
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0041859, 0.0042196

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036358, upper bound: 0.0035735
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036368, upper bound: 0.0035700
time: 1.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022684, 0.0022815
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057563, 0.0057895
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035712, 0.0035919
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0067069, 0.0066684
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058551, 0.0058889
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022178, 0.0022306
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0085119, 0.0084631
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059562, 0.0059220
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063860, 0.0063494
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0041941, 0.0042183

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036302, upper bound: 0.0035807
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036305, upper bound: 0.0035777
time: 1.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022747, 0.0022720
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057723, 0.0057656
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035812, 0.0035770
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066792, 0.0066870
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058714, 0.0058646
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022239, 0.0022214
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084768, 0.0084866
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059317, 0.0059385
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063597, 0.0063670
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042058, 0.0042009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036003, upper bound: 0.0036135
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036009, upper bound: 0.0036102
time: 1.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022791, 0.0022714
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057837, 0.0057639
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035882, 0.0035760
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066772, 0.0067001
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058830, 0.0058629
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022283, 0.0022207
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084743, 0.0085033
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059299, 0.0059502
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063578, 0.0063795
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042140, 0.0041997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035935, upper bound: 0.0036213
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035952, upper bound: 0.0036185
time: 1.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022639, 0.0022826
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057449, 0.0057925
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035641, 0.0035937
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0067103, 0.0066551
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058435, 0.0058919
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022134, 0.0022317
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0085162, 0.0084462
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059593, 0.0059103
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063893, 0.0063367
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0041858, 0.0042205

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036317, upper bound: 0.0035746
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036360, upper bound: 0.0035731
time: 1.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022683, 0.0022818
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057562, 0.0057903
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035712, 0.0035924
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0067078, 0.0066683
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058550, 0.0058898
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022177, 0.0022309
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0085131, 0.0084629
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059571, 0.0059219
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063869, 0.0063492
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0041940, 0.0042189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036295, upper bound: 0.0035830
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036336, upper bound: 0.0035818
time: 1.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022748, 0.0022723
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057727, 0.0057664
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035814, 0.0035775
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066801, 0.0066874
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058718, 0.0058654
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022241, 0.0022216
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084779, 0.0084872
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059324, 0.0059389
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063605, 0.0063674
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042061, 0.0042015

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035951, upper bound: 0.0036146
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035981, upper bound: 0.0036125
time: 1.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022793, 0.0022715
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057840, 0.0057642
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035884, 0.0035761
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066775, 0.0067005
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058833, 0.0058631
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022284, 0.0022208
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084746, 0.0085038
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059301, 0.0059506
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063580, 0.0063799
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042143, 0.0041998

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035898, upper bound: 0.0036246
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035934, upper bound: 0.0036228
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022667, 0.0022791
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057521, 0.0057835
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035686, 0.0035881
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066999, 0.0066635
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058508, 0.0058828
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022161, 0.0022282
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0085031, 0.0084569
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059500, 0.0059177
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063794, 0.0063447
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0041910, 0.0042139

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036249, upper bound: 0.0035746
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036302, upper bound: 0.0035730
time: 1.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022712, 0.0022785
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057634, 0.0057820
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035756, 0.0035871
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066981, 0.0066766
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058624, 0.0058812
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022205, 0.0022276
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0085008, 0.0084735
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059484, 0.0059294
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063777, 0.0063572
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0041993, 0.0042128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036205, upper bound: 0.0035829
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036250, upper bound: 0.0035816
time: 1.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022778, 0.0022690
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057802, 0.0057580
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035861, 0.0035723
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066703, 0.0066961
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058794, 0.0058568
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022270, 0.0022184
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084655, 0.0084982
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059238, 0.0059466
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063512, 0.0063757
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042115, 0.0041953

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035936, upper bound: 0.0036166
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035980, upper bound: 0.0036155
time: 1.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022822, 0.0022684
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057915, 0.0057565
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035931, 0.0035714
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066686, 0.0067092
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058910, 0.0058553
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022313, 0.0022178
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084634, 0.0085148
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059222, 0.0059583
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063496, 0.0063882
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042198, 0.0041943

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035883, upper bound: 0.0036272
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035931, upper bound: 0.0036271
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022712, 0.0022753
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057636, 0.0057739
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035757, 0.0035821
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066888, 0.0066768
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058625, 0.0058730
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022206, 0.0022245
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084889, 0.0084738
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059401, 0.0059295
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063688, 0.0063574
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0041994, 0.0042069

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036238, upper bound: 0.0035979
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036238, upper bound: 0.0035938
time: 1.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022757, 0.0022746
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057749, 0.0057721
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035828, 0.0035811
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066867, 0.0066899
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058740, 0.0058712
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022249, 0.0022239
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084863, 0.0084904
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059383, 0.0059412
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063668, 0.0063699
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042077, 0.0042056

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036138, upper bound: 0.0036041
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036145, upper bound: 0.0036001
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022818, 0.0022648
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057904, 0.0057473
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035924, 0.0035656
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066579, 0.0067079
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058898, 0.0058459
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022309, 0.0022143
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084498, 0.0085132
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059127, 0.0059571
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063394, 0.0063870
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042190, 0.0041875

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035811, upper bound: 0.0036360
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035823, upper bound: 0.0036310
time: 1.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022863, 0.0022638
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0058017, 0.0057448
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035994, 0.0035641
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066550, 0.0067210
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0059013, 0.0058434
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022353, 0.0022133
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084461, 0.0085299
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059102, 0.0059688
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063366, 0.0063995
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042272, 0.0041857

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035728, upper bound: 0.0036406
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035745, upper bound: 0.0036364
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022745, 0.0022726
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057719, 0.0057671
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035809, 0.0035779
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066809, 0.0066865
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058710, 0.0058661
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022238, 0.0022219
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084789, 0.0084860
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059331, 0.0059381
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063612, 0.0063666
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042055, 0.0042020

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036182, upper bound: 0.0035980
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036193, upper bound: 0.0035948
time: 1.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022790, 0.0022717
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057832, 0.0057648
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035879, 0.0035765
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066782, 0.0066996
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058825, 0.0058637
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022281, 0.0022210
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084755, 0.0085027
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059307, 0.0059497
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063587, 0.0063791
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042137, 0.0042003

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036092, upper bound: 0.0036041
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036103, upper bound: 0.0036005
time: 1.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022853, 0.0022622
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057992, 0.0057406
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035978, 0.0035615
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066502, 0.0067181
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058988, 0.0058391
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022343, 0.0022117
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084400, 0.0085261
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059059, 0.0059662
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063320, 0.0063967
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042254, 0.0041827

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035811, upper bound: 0.0036422
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035824, upper bound: 0.0036393
time: 1.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022897, 0.0022612
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0058105, 0.0057382
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036049, 0.0035600
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066474, 0.0067312
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0059103, 0.0058367
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022387, 0.0022108
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084364, 0.0085428
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059034, 0.0059778
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063294, 0.0064092
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042336, 0.0041809

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035728, upper bound: 0.0036462
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035745, upper bound: 0.0036435
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022742, 0.0022722
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057711, 0.0057661
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035804, 0.0035773
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066797, 0.0066855
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058702, 0.0058651
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022235, 0.0022215
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084774, 0.0084848
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059321, 0.0059373
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063601, 0.0063657
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042049, 0.0042012

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036146, upper bound: 0.0036002
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036178, upper bound: 0.0035989
time: 1.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022787, 0.0022714
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057824, 0.0057641
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035874, 0.0035761
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066774, 0.0066987
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058817, 0.0058630
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022278, 0.0022208
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084745, 0.0085015
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059301, 0.0059489
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063580, 0.0063782
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042131, 0.0041998

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036073, upper bound: 0.0036069
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036114, upper bound: 0.0036060
time: 1.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022851, 0.0022621
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057987, 0.0057403
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035975, 0.0035613
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066499, 0.0067175
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058982, 0.0058389
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022341, 0.0022116
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084395, 0.0085253
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059056, 0.0059656
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063317, 0.0063961
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042250, 0.0041825

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035769, upper bound: 0.0036408
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035798, upper bound: 0.0036397
time: 1.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022895, 0.0022610
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0058100, 0.0057377
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036045, 0.0035597
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066469, 0.0067306
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0059097, 0.0058362
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022384, 0.0022106
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084358, 0.0085420
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059029, 0.0059773
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063289, 0.0064086
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042332, 0.0041806

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035698, upper bound: 0.0036477
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035735, upper bound: 0.0036470
time: 1.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022775, 0.0022695
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057795, 0.0057591
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035856, 0.0035730
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066717, 0.0066952
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058787, 0.0058580
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022267, 0.0022189
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084672, 0.0084971
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059250, 0.0059459
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063525, 0.0063749
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042110, 0.0041962

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036091, upper bound: 0.0036002
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036147, upper bound: 0.0035989
time: 1.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022820, 0.0022686
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0057908, 0.0057568
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0035926, 0.0035715
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066689, 0.0067084
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0058902, 0.0058556
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022311, 0.0022179
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084638, 0.0085138
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0059225, 0.0059575
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063499, 0.0063874
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042192, 0.0041945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036021, upper bound: 0.0036067
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036083, upper bound: 0.0036059
time: 1.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022885, 0.0022591
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0058074, 0.0057328
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036029, 0.0035567
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066412, 0.0067276
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0059071, 0.0058312
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022374, 0.0022087
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084286, 0.0085382
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0058979, 0.0059746
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063235, 0.0064057
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042313, 0.0041770

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035768, upper bound: 0.0036463
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035799, upper bound: 0.0036463
time: 1.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022930, 0.0022582
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0058187, 0.0057305
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0036099, 0.0035552
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0066385, 0.0067407
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0059186, 0.0058289
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0022418, 0.0022078
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0084251, 0.0085548
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0058955, 0.0059862
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0063209, 0.0064182
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0042396, 0.0041753

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0035696, upper bound: 0.0036539
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0035735, upper bound: 0.0036540
time: 1.14 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 4.63 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036540, upper bound: 0.0035735
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036539, upper bound: 0.0035695
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036463, upper bound: 0.0035799
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036463, upper bound: 0.0035768
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036058, upper bound: 0.0036083
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036067, upper bound: 0.0036021
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035989, upper bound: 0.0036147
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036001, upper bound: 0.0036091
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036470, upper bound: 0.0035735
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036478, upper bound: 0.0035697
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036397, upper bound: 0.0035798
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036408, upper bound: 0.0035768
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036060, upper bound: 0.0036113
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036068, upper bound: 0.0036074
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035989, upper bound: 0.0036178
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036001, upper bound: 0.0036146
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036435, upper bound: 0.0035745
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036463, upper bound: 0.0035727
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036393, upper bound: 0.0035824
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036422, upper bound: 0.0035812
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036005, upper bound: 0.0036105
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036040, upper bound: 0.0036092
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035948, upper bound: 0.0036193
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035979, upper bound: 0.0036182
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036365, upper bound: 0.0035746
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036406, upper bound: 0.0035727
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036310, upper bound: 0.0035823
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036360, upper bound: 0.0035811
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036001, upper bound: 0.0036146
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036039, upper bound: 0.0036139
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035938, upper bound: 0.0036238
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035978, upper bound: 0.0036238
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036271, upper bound: 0.0035931
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036271, upper bound: 0.0035883
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036153, upper bound: 0.0035980
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036165, upper bound: 0.0035936
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035816, upper bound: 0.0036251
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035829, upper bound: 0.0036205
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035731, upper bound: 0.0036302
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035746, upper bound: 0.0036249
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036228, upper bound: 0.0035934
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036245, upper bound: 0.0035898
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036126, upper bound: 0.0035981
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036146, upper bound: 0.0035949
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035818, upper bound: 0.0036336
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035830, upper bound: 0.0036295
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035731, upper bound: 0.0036361
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035746, upper bound: 0.0036317
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036185, upper bound: 0.0035951
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036213, upper bound: 0.0035935
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036103, upper bound: 0.0036010
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035735, upper bound: 0.0036004
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035777, upper bound: 0.0036305
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035807, upper bound: 0.0036302
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035700, upper bound: 0.0036368
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035735, upper bound: 0.0036359
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036137, upper bound: 0.0035952
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036189, upper bound: 0.0035940
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036065, upper bound: 0.0036009
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036119, upper bound: 0.0036006
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035776, upper bound: 0.0036390
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035807, upper bound: 0.0036387
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035700, upper bound: 0.0036437
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035737, upper bound: 0.0036436
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036436, upper bound: 0.0035737
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036437, upper bound: 0.0035700
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036387, upper bound: 0.0035807
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036390, upper bound: 0.0035776
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036004, upper bound: 0.0036119
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036009, upper bound: 0.0036065
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035940, upper bound: 0.0036188
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035952, upper bound: 0.0036137
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036358, upper bound: 0.0035735
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036368, upper bound: 0.0035700
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036302, upper bound: 0.0035807
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036305, upper bound: 0.0035777
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036003, upper bound: 0.0036135
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036009, upper bound: 0.0036102
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035935, upper bound: 0.0036213
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035952, upper bound: 0.0036185
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036317, upper bound: 0.0035746
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036360, upper bound: 0.0035731
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036295, upper bound: 0.0035830
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036336, upper bound: 0.0035818
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035951, upper bound: 0.0036146
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035981, upper bound: 0.0036125
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035898, upper bound: 0.0036246
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035934, upper bound: 0.0036228
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036249, upper bound: 0.0035746
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036302, upper bound: 0.0035730
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036205, upper bound: 0.0035829
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036250, upper bound: 0.0035816
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035936, upper bound: 0.0036166
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035980, upper bound: 0.0036155
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035883, upper bound: 0.0036272
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035931, upper bound: 0.0036271
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036238, upper bound: 0.0035979
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036238, upper bound: 0.0035938
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036138, upper bound: 0.0036041
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036145, upper bound: 0.0036001
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035811, upper bound: 0.0036360
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035823, upper bound: 0.0036310
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035728, upper bound: 0.0036406
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035745, upper bound: 0.0036364
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036182, upper bound: 0.0035980
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036193, upper bound: 0.0035948
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036092, upper bound: 0.0036041
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036103, upper bound: 0.0036005
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035811, upper bound: 0.0036422
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035824, upper bound: 0.0036393
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035728, upper bound: 0.0036462
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035745, upper bound: 0.0036435
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036146, upper bound: 0.0036002
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036178, upper bound: 0.0035989
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036073, upper bound: 0.0036069
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036114, upper bound: 0.0036060
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035769, upper bound: 0.0036408
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035798, upper bound: 0.0036397
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035698, upper bound: 0.0036477
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035735, upper bound: 0.0036470
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036091, upper bound: 0.0036002
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036147, upper bound: 0.0035989
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036021, upper bound: 0.0036067
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0036083, upper bound: 0.0036059
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035768, upper bound: 0.0036463
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035799, upper bound: 0.0036463
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035696, upper bound: 0.0036539
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 7, lower bound: -0.0035735, upper bound: 0.0036540

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0021907, 0.0022191
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0055592, 0.0056313
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0034490, 0.0034937
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0065236, 0.0064401
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0056547, 0.0057280
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0021418, 0.0021696
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0082793, 0.0081733
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0057934, 0.0057193
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0062115, 0.0061320
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0040505, 0.0041030

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035989, upper bound: 0.0034489
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035295, upper bound: 0.0035166
time: 1.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0021879, 0.0022171
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0055520, 0.0056262
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0034445, 0.0034905
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0065176, 0.0064317
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0056473, 0.0057227
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0021390, 0.0021676
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0082717, 0.0081627
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0057881, 0.0057118
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0062058, 0.0061240
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0040452, 0.0040993

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035989, upper bound: 0.0034424
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035342, upper bound: 0.0035141
time: 1.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022210, 0.0021879
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0056362, 0.0055520
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0034967, 0.0034445
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0064317, 0.0065293
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0057330, 0.0056473
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0021715, 0.0021390
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0081627, 0.0082865
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0057118, 0.0057985
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0061240, 0.0062169
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0041066, 0.0040452

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035141, upper bound: 0.0035343
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0034424, upper bound: 0.0035990
time: 1.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0033021, -0.0004227, -0.0033021, -0.0004227, -0.0022226, 0.0021907
1: -0.0126902, -0.0053831, -0.0126902, -0.0053831, -0.0056402, 0.0055592
2: 0.0271569, 0.0316903, 0.0271569, 0.0316903, -0.0034992, 0.0034490
3: 0.0003887, 0.0088536, 0.0003887, 0.0088536, -0.0064401, 0.0065339
4: -0.0118011, -0.0043686, -0.0118011, -0.0043686, -0.0057370, 0.0056547
5: 0.0092682, 0.0120835, 0.0092682, 0.0120835, -0.0021730, 0.0021418
6: 0.0008814, 0.0116245, 0.0008814, 0.0116245, -0.0081733, 0.0082924
7: 0.9786761, 0.9861935, 0.9786761, 0.9861935, -0.0057193, 0.0058026
8: -0.0094269, -0.0013669, -0.0094269, -0.0013669, -0.0061320, 0.0062213
9: -0.0040967, 0.0012274, -0.0040967, 0.0012274, -0.0041095, 0.0040505

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035166, upper bound: 0.0035295
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0034489, upper bound: 0.0035989
time: 1.51 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 4.70 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.70
Output dim: 7, lower bound: -0.0035989, upper bound: 0.0034489
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.70
Output dim: 7, lower bound: -0.0035295, upper bound: 0.0035166
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.70
Output dim: 7, lower bound: -0.0035989, upper bound: 0.0034424
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.70
Output dim: 7, lower bound: -0.0035342, upper bound: 0.0035141
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.70
Output dim: 7, lower bound: -0.0035141, upper bound: 0.0035343
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.70
Output dim: 7, lower bound: -0.0034424, upper bound: 0.0035990
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.70
Output dim: 7, lower bound: -0.0035166, upper bound: 0.0035295
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.70
Output dim: 7, lower bound: -0.0034489, upper bound: 0.0035989

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.87 + 590.10 = 593.97 seconds
