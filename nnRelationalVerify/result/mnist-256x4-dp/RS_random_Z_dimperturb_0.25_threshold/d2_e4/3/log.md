## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0071008


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239)
1: (0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0153637, 0.0153637)
2: (-0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561)
3: (-0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0028109, 0.0028109)
4: (-0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834)
5: (-0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360)
6: (-0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510)
7: (-0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640)
8: (-0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131)
9: (-0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.57 + 1.78 = 3.35 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0132262, upper bound: 0.0132262

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0129190, upper bound: 0.0129190
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0129190, upper bound: 0.0129190
time: 0.96 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.89 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.89
Output dim: 1, lower bound: -0.0129190, upper bound: 0.0129190
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.89
Output dim: 1, lower bound: -0.0129190, upper bound: 0.0129190

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0153612, 0.0153623
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0028103, 0.0028099
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121010, upper bound: 0.0121019
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121019, upper bound: 0.0121010
time: 0.88 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0153623, 0.0153612
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0028099, 0.0028103
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0125742, upper bound: 0.0125850
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0125850, upper bound: 0.0125742
time: 0.82 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.19 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.19
Output dim: 1, lower bound: -0.0121010, upper bound: 0.0121019
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.19
Output dim: 1, lower bound: -0.0121019, upper bound: 0.0121010
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.19
Output dim: 1, lower bound: -0.0125742, upper bound: 0.0125850
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.19
Output dim: 1, lower bound: -0.0125850, upper bound: 0.0125742

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152919, 0.0153175
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027928, 0.0027821
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0109799, upper bound: 0.0109799
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0109799, upper bound: 0.0109799
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0153164, 0.0152928
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027825, 0.0027923
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0117911, upper bound: 0.0117911
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0117911, upper bound: 0.0117912
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0153515, 0.0153529
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0028069, 0.0028063
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0124195, upper bound: 0.0124297
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0124180, upper bound: 0.0124318
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0153540, 0.0153504
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0028058, 0.0028073
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0125659, upper bound: 0.0125609
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0125724, upper bound: 0.0125599
time: 0.90 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.23 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 1, lower bound: -0.0109799, upper bound: 0.0109799
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 1, lower bound: -0.0109799, upper bound: 0.0109799
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 1, lower bound: -0.0117911, upper bound: 0.0117911
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 1, lower bound: -0.0117911, upper bound: 0.0117912
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 1, lower bound: -0.0124195, upper bound: 0.0124297
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 1, lower bound: -0.0124180, upper bound: 0.0124318
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 1, lower bound: -0.0125659, upper bound: 0.0125609
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 1, lower bound: -0.0125724, upper bound: 0.0125599

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152820, 0.0153073
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027881, 0.0027776
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0104134, upper bound: 0.0104134
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0104134, upper bound: 0.0104134
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152817, 0.0153175
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027928, 0.0027775
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0108389, upper bound: 0.0108218
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0108218, upper bound: 0.0108389
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152697, 0.0152096
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027529, 0.0027778
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0115891, upper bound: 0.0115898
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0115900, upper bound: 0.0115879
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152333, 0.0152478
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027688, 0.0027627
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0116488, upper bound: 0.0116055
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0116055, upper bound: 0.0116487
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152781, 0.0152820
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027779, 0.0027763
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0116352, upper bound: 0.0116317
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0116352, upper bound: 0.0116317
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152805, 0.0152793
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027768, 0.0027773
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121814, upper bound: 0.0122040
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121934, upper bound: 0.0121919
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0153530, 0.0153519
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0028061, 0.0028066
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0125382, upper bound: 0.0125511
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0125559, upper bound: 0.0125269
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0153555, 0.0153498
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0028052, 0.0028076
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0115334, upper bound: 0.0115303
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0115334, upper bound: 0.0115303
time: 0.91 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.25 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 1, lower bound: -0.0104134, upper bound: 0.0104134
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 1, lower bound: -0.0104134, upper bound: 0.0104134
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 1, lower bound: -0.0108389, upper bound: 0.0108218
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 1, lower bound: -0.0108218, upper bound: 0.0108389
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 1, lower bound: -0.0115891, upper bound: 0.0115898
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 1, lower bound: -0.0115900, upper bound: 0.0115879
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 1, lower bound: -0.0116488, upper bound: 0.0116055
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 1, lower bound: -0.0116055, upper bound: 0.0116487
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 1, lower bound: -0.0116352, upper bound: 0.0116317
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 1, lower bound: -0.0116352, upper bound: 0.0116317
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 1, lower bound: -0.0121814, upper bound: 0.0122040
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 1, lower bound: -0.0121934, upper bound: 0.0121919
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 1, lower bound: -0.0125382, upper bound: 0.0125511
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 1, lower bound: -0.0125559, upper bound: 0.0125269
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 1, lower bound: -0.0115334, upper bound: 0.0115303
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 1, lower bound: -0.0115334, upper bound: 0.0115303

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152716, 0.0152990
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027853, 0.0027739
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100789, upper bound: 0.0100874
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100874, upper bound: 0.0100789
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152737, 0.0152965
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027842, 0.0027748
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0101844, upper bound: 0.0101874
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0101874, upper bound: 0.0101844
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152065, 0.0152467
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027637, 0.0027463
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0108297, upper bound: 0.0108142
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0108310, upper bound: 0.0108111
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152109, 0.0152442
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027626, 0.0027482
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0108070, upper bound: 0.0108315
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0108182, upper bound: 0.0108185
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152229, 0.0151725
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027319, 0.0027528
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0111027, upper bound: 0.0111049
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0111027, upper bound: 0.0111049
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152326, 0.0151632
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027280, 0.0027568
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0103933, upper bound: 0.0103933
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0103933, upper bound: 0.0103933
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0150470, 0.0150490
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0026783, 0.0026775
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100774, upper bound: 0.0100742
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100774, upper bound: 0.0100742
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0150345, 0.0150563
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0026813, 0.0026723
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0111343, upper bound: 0.0111691
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0111378, upper bound: 0.0111569
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152386, 0.0152468
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027621, 0.0027587
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0110384, upper bound: 0.0110246
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0110384, upper bound: 0.0110246
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152430, 0.0152408
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027596, 0.0027605
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0110397, upper bound: 0.0110327
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0110404, upper bound: 0.0110327
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152346, 0.0152436
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027550, 0.0027513
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0111581, upper bound: 0.0111885
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0111581, upper bound: 0.0111885
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152449, 0.0152345
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027512, 0.0027556
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0118356, upper bound: 0.0118460
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0118355, upper bound: 0.0118460
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0153739, 0.0153822
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0028222, 0.0028188
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122002, upper bound: 0.0122108
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122002, upper bound: 0.0122108
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0153837, 0.0153728
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0028183, 0.0028228
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0116828, upper bound: 0.0116617
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0116828, upper bound: 0.0116617
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0153455, 0.0153401
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0028006, 0.0028029
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0113857, upper bound: 0.0113759
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0113786, upper bound: 0.0113824
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0153458, 0.0153498
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0028052, 0.0028030
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0112274, upper bound: 0.0112442
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0112459, upper bound: 0.0112257
time: 0.93 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.14 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 1, lower bound: -0.0100789, upper bound: 0.0100874
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 1, lower bound: -0.0100874, upper bound: 0.0100789
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 1, lower bound: -0.0101844, upper bound: 0.0101874
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 1, lower bound: -0.0101874, upper bound: 0.0101844
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 1, lower bound: -0.0108297, upper bound: 0.0108142
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 1, lower bound: -0.0108310, upper bound: 0.0108111
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 1, lower bound: -0.0108070, upper bound: 0.0108315
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 1, lower bound: -0.0108182, upper bound: 0.0108185
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 1, lower bound: -0.0111027, upper bound: 0.0111049
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 1, lower bound: -0.0111027, upper bound: 0.0111049
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 1, lower bound: -0.0103933, upper bound: 0.0103933
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 1, lower bound: -0.0103933, upper bound: 0.0103933
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 1, lower bound: -0.0100774, upper bound: 0.0100742
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 1, lower bound: -0.0100774, upper bound: 0.0100742
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 1, lower bound: -0.0111343, upper bound: 0.0111691
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 1, lower bound: -0.0111378, upper bound: 0.0111569
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 1, lower bound: -0.0110384, upper bound: 0.0110246
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 1, lower bound: -0.0110384, upper bound: 0.0110246
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 1, lower bound: -0.0110397, upper bound: 0.0110327
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 1, lower bound: -0.0110404, upper bound: 0.0110327
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 1, lower bound: -0.0111581, upper bound: 0.0111885
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 1, lower bound: -0.0111581, upper bound: 0.0111885
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 1, lower bound: -0.0118356, upper bound: 0.0118460
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 1, lower bound: -0.0118355, upper bound: 0.0118460
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 1, lower bound: -0.0122002, upper bound: 0.0122108
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 1, lower bound: -0.0122002, upper bound: 0.0122108
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 1, lower bound: -0.0116828, upper bound: 0.0116617
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 1, lower bound: -0.0116828, upper bound: 0.0116617
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 1, lower bound: -0.0113857, upper bound: 0.0113759
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 1, lower bound: -0.0113786, upper bound: 0.0113824
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 1, lower bound: -0.0112274, upper bound: 0.0112442
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 1, lower bound: -0.0112459, upper bound: 0.0112257

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151986, 0.0152424
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027638, 0.0027457
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0098651, upper bound: 0.0098772
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0098692, upper bound: 0.0098676
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152171, 0.0152260
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027570, 0.0027533
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091687, upper bound: 0.0091689
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091687, upper bound: 0.0091689
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152264, 0.0152589
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027635, 0.0027500
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0098259, upper bound: 0.0098384
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0098259, upper bound: 0.0098384
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152362, 0.0152486
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027592, 0.0027540
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0092264, upper bound: 0.0092257
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0092264, upper bound: 0.0092257
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152194, 0.0152687
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027759, 0.0027548
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0106934, upper bound: 0.0106765
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0106935, upper bound: 0.0106765
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152326, 0.0152594
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027721, 0.0027603
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097979, upper bound: 0.0097700
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097979, upper bound: 0.0097700
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152095, 0.0152442
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027622, 0.0027474
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0106131, upper bound: 0.0106419
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0106135, upper bound: 0.0106418
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152109, 0.0152424
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027615, 0.0027480
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100894, upper bound: 0.0100840
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100894, upper bound: 0.0100840
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152120, 0.0151642
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027291, 0.0027490
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0109564, upper bound: 0.0109255
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0109215, upper bound: 0.0109584
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152146, 0.0151620
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027282, 0.0027500
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0104120, upper bound: 0.0104126
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0104120, upper bound: 0.0104126
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152280, 0.0151572
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027255, 0.0027548
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0101884, upper bound: 0.0101885
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0101884, upper bound: 0.0101885
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152266, 0.0151585
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027260, 0.0027543
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097127, upper bound: 0.0097141
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097127, upper bound: 0.0097141
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0150368, 0.0150390
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0026735, 0.0026726
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0098092, upper bound: 0.0098068
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0098092, upper bound: 0.0098068
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0150369, 0.0150490
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0026783, 0.0026727
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0098092, upper bound: 0.0098068
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0098092, upper bound: 0.0098068
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0149586, 0.0149988
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0026586, 0.0026420
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0109934, upper bound: 0.0110121
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0109856, upper bound: 0.0110321
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0149749, 0.0149804
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0026510, 0.0026487
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0108182, upper bound: 0.0108353
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0108182, upper bound: 0.0108353
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151886, 0.0151616
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027322, 0.0027434
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0110363, upper bound: 0.0110246
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0110371, upper bound: 0.0110238
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151534, 0.0151980
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027473, 0.0027288
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0081115, upper bound: 0.0081115
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0081115, upper bound: 0.0081115
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151735, 0.0151972
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027427, 0.0027328
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0099715, upper bound: 0.0099681
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0099715, upper bound: 0.0099681
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151994, 0.0151730
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027326, 0.0027436
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0104527, upper bound: 0.0104445
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0104527, upper bound: 0.0104445
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152119, 0.0152135
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027424, 0.0027418
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0108135, upper bound: 0.0108723
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0108422, upper bound: 0.0108459
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152045, 0.0152436
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027550, 0.0027387
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0105654, upper bound: 0.0105789
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0105654, upper bound: 0.0105789
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151956, 0.0151490
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027212, 0.0027405
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0105735, upper bound: 0.0105745
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0105735, upper bound: 0.0105745
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151594, 0.0151857
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027364, 0.0027255
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0106729, upper bound: 0.0106734
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0106729, upper bound: 0.0106734
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0153296, 0.0153004
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027930, 0.0028052
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0120611, upper bound: 0.0120319
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0120100, upper bound: 0.0120705
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152922, 0.0153396
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0028093, 0.0027896
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0120611, upper bound: 0.0120319
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0120100, upper bound: 0.0120705
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0153791, 0.0153670
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0028158, 0.0028209
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0111265, upper bound: 0.0111149
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0111265, upper bound: 0.0111149
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0153779, 0.0153682
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0028163, 0.0028203
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0105763, upper bound: 0.0105570
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0105763, upper bound: 0.0105570
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152691, 0.0152673
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027712, 0.0027719
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0107694, upper bound: 0.0107611
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0107694, upper bound: 0.0107609
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152727, 0.0152651
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027703, 0.0027735
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0107620, upper bound: 0.0107673
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0107624, upper bound: 0.0107673
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152728, 0.0152934
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027834, 0.0027743
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096930, upper bound: 0.0096816
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096930, upper bound: 0.0096816
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152892, 0.0152768
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027766, 0.0027811
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0110902, upper bound: 0.0110580
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0110707, upper bound: 0.0110687
time: 0.89 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.55 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0098651, upper bound: 0.0098772
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0098692, upper bound: 0.0098676
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0091687, upper bound: 0.0091689
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0091687, upper bound: 0.0091689
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0098259, upper bound: 0.0098384
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0098259, upper bound: 0.0098384
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0092264, upper bound: 0.0092257
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0092264, upper bound: 0.0092257
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0106934, upper bound: 0.0106765
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0106935, upper bound: 0.0106765
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0097979, upper bound: 0.0097700
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0097979, upper bound: 0.0097700
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0106131, upper bound: 0.0106419
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0106135, upper bound: 0.0106418
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0100894, upper bound: 0.0100840
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0100894, upper bound: 0.0100840
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0109564, upper bound: 0.0109255
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0109215, upper bound: 0.0109584
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0104120, upper bound: 0.0104126
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0104120, upper bound: 0.0104126
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0101884, upper bound: 0.0101885
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0101884, upper bound: 0.0101885
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0097127, upper bound: 0.0097141
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0097127, upper bound: 0.0097141
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0098092, upper bound: 0.0098068
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0098092, upper bound: 0.0098068
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0098092, upper bound: 0.0098068
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0098092, upper bound: 0.0098068
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0109934, upper bound: 0.0110121
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0109856, upper bound: 0.0110321
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0108182, upper bound: 0.0108353
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0108182, upper bound: 0.0108353
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0110363, upper bound: 0.0110246
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0110371, upper bound: 0.0110238
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0081115, upper bound: 0.0081115
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0081115, upper bound: 0.0081115
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0099715, upper bound: 0.0099681
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0099715, upper bound: 0.0099681
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0104527, upper bound: 0.0104445
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0104527, upper bound: 0.0104445
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0108135, upper bound: 0.0108723
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0108422, upper bound: 0.0108459
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0105654, upper bound: 0.0105789
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0105654, upper bound: 0.0105789
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0105735, upper bound: 0.0105745
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0105735, upper bound: 0.0105745
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0106729, upper bound: 0.0106734
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0106729, upper bound: 0.0106734
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0120611, upper bound: 0.0120319
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0120100, upper bound: 0.0120705
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0120611, upper bound: 0.0120319
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0120100, upper bound: 0.0120705
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0111265, upper bound: 0.0111149
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0111265, upper bound: 0.0111149
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0105763, upper bound: 0.0105570
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0105763, upper bound: 0.0105570
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0107694, upper bound: 0.0107611
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0107694, upper bound: 0.0107609
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0107620, upper bound: 0.0107673
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0107624, upper bound: 0.0107673
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0096930, upper bound: 0.0096816
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0096930, upper bound: 0.0096816
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0110902, upper bound: 0.0110580
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 1, lower bound: -0.0110707, upper bound: 0.0110687

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151531, 0.0152069
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027434, 0.0027211
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096142, upper bound: 0.0096335
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096142, upper bound: 0.0096335
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151631, 0.0151964
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027390, 0.0027252
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089434, upper bound: 0.0089398
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089434, upper bound: 0.0089398
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151766, 0.0151907
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027421, 0.0027363
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090379, upper bound: 0.0090199
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090199, upper bound: 0.0090383
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151818, 0.0151847
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027396, 0.0027384
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090379, upper bound: 0.0090199
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090199, upper bound: 0.0090383
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152018, 0.0152433
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027572, 0.0027399
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096142, upper bound: 0.0096335
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096148, upper bound: 0.0096274
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152095, 0.0152343
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027534, 0.0027431
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088870, upper bound: 0.0088872
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088870, upper bound: 0.0088872
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151943, 0.0152133
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027439, 0.0027360
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088872, upper bound: 0.0088870
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088872, upper bound: 0.0088870
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152009, 0.0152097
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027423, 0.0027387
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090946, upper bound: 0.0090820
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090851, upper bound: 0.0090938
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151937, 0.0152523
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027691, 0.0027441
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0098972, upper bound: 0.0098935
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0098972, upper bound: 0.0098935
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152045, 0.0152430
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027652, 0.0027485
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0106743, upper bound: 0.0106736
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0106896, upper bound: 0.0106578
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151960, 0.0152239
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027575, 0.0027452
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097892, upper bound: 0.0097700
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097979, upper bound: 0.0097638
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151971, 0.0152179
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027550, 0.0027457
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093132, upper bound: 0.0093040
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093137, upper bound: 0.0093039
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151646, 0.0152087
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027413, 0.0027226
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095698, upper bound: 0.0096057
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095698, upper bound: 0.0096057
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151740, 0.0151996
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027376, 0.0027265
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095708, upper bound: 0.0096051
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095708, upper bound: 0.0096051
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151646, 0.0151581
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027310, 0.0027333
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0098217, upper bound: 0.0098123
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0098217, upper bound: 0.0098123
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151265, 0.0151943
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027461, 0.0027175
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0098217, upper bound: 0.0098123
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0098217, upper bound: 0.0098123
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0150213, 0.0149619
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0026366, 0.0026612
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095465, upper bound: 0.0095355
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095465, upper bound: 0.0095355
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0150097, 0.0149702
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0026400, 0.0026564
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0107696, upper bound: 0.0107853
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0107613, upper bound: 0.0108107
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151716, 0.0151257
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027130, 0.0027321
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100754, upper bound: 0.0100786
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100782, upper bound: 0.0100759
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151783, 0.0151203
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027108, 0.0027348
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0104109, upper bound: 0.0104126
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0104120, upper bound: 0.0104110
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152033, 0.0151408
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027188, 0.0027447
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0101748, upper bound: 0.0101775
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0101775, upper bound: 0.0101753
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152126, 0.0151325
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027154, 0.0027486
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093146, upper bound: 0.0093175
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093146, upper bound: 0.0093175
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151558, 0.0151051
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027053, 0.0027263
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097019, upper bound: 0.0097029
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097019, upper bound: 0.0097032
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151712, 0.0150877
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0026981, 0.0027327
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097019, upper bound: 0.0097029
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097019, upper bound: 0.0097032
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0150122, 0.0150222
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0026666, 0.0026625
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093726, upper bound: 0.0093765
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093766, upper bound: 0.0093727
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0150222, 0.0150144
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0026634, 0.0026666
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093726, upper bound: 0.0093765
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093766, upper bound: 0.0093727
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0150123, 0.0150323
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0026714, 0.0026625
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097990, upper bound: 0.0098068
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0098092, upper bound: 0.0097959
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0150222, 0.0150245
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0026682, 0.0026666
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097830, upper bound: 0.0097959
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097982, upper bound: 0.0097768
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0148922, 0.0149351
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0026331, 0.0026153
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0105442, upper bound: 0.0105564
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0105442, upper bound: 0.0105564
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0148949, 0.0149387
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0026345, 0.0026164
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0109697, upper bound: 0.0110234
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0109771, upper bound: 0.0110196
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0149494, 0.0149635
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0026444, 0.0026386
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0108156, upper bound: 0.0108353
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0108182, upper bound: 0.0108262
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0149595, 0.0149549
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0026409, 0.0026428
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0104305, upper bound: 0.0104506
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0104305, upper bound: 0.0104506
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151873, 0.0151623
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027322, 0.0027425
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0108203, upper bound: 0.0108119
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0108237, upper bound: 0.0108080
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151893, 0.0151600
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027312, 0.0027434
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100790, upper bound: 0.0100697
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100790, upper bound: 0.0100697
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151418, 0.0151878
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027424, 0.0027234
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0080903, upper bound: 0.0080991
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0080990, upper bound: 0.0080882
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151432, 0.0151980
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027473, 0.0027239
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0080938, upper bound: 0.0081115
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0081115, upper bound: 0.0080876
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151689, 0.0151916
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027403, 0.0027308
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097478, upper bound: 0.0097490
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097515, upper bound: 0.0097478
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151678, 0.0151926
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027407, 0.0027304
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096574, upper bound: 0.0096524
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096574, upper bound: 0.0096524
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151498, 0.0150867
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027027, 0.0027289
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0101369, upper bound: 0.0101172
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0101370, upper bound: 0.0101134
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151131, 0.0151234
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027179, 0.0027136
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0101369, upper bound: 0.0101172
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0101370, upper bound: 0.0101134
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151324, 0.0151510
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027168, 0.0027091
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0107996, upper bound: 0.0108590
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0107999, upper bound: 0.0108575
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151470, 0.0151340
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027097, 0.0027151
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0092813, upper bound: 0.0092973
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0092813, upper bound: 0.0092973
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151559, 0.0151581
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027249, 0.0027241
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0105524, upper bound: 0.0105673
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0105538, upper bound: 0.0105620
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151191, 0.0151948
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027402, 0.0027088
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0105524, upper bound: 0.0105673
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0105538, upper bound: 0.0105620
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151715, 0.0151189
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027088, 0.0027306
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0104161, upper bound: 0.0104056
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0103999, upper bound: 0.0104177
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151655, 0.0151490
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027212, 0.0027281
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100421, upper bound: 0.0100728
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100664, upper bound: 0.0100572
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151494, 0.0151755
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027315, 0.0027207
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0101000, upper bound: 0.0101137
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0101010, upper bound: 0.0101137
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151493, 0.0151857
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027364, 0.0027206
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0106567, upper bound: 0.0106636
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0106614, upper bound: 0.0106629
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151333, 0.0150957
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027036, 0.0027192
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0108035, upper bound: 0.0107923
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0108035, upper bound: 0.0107923
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151249, 0.0151079
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027086, 0.0027157
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0107884, upper bound: 0.0108086
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0107884, upper bound: 0.0108086
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151007, 0.0151349
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027198, 0.0027056
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0118941, upper bound: 0.0118626
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0118774, upper bound: 0.0118687
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0150874, 0.0151424
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027229, 0.0027001
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0117867, upper bound: 0.0118511
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0117902, upper bound: 0.0118459
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0153377, 0.0152852
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027867, 0.0028085
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0102275, upper bound: 0.0102179
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0102275, upper bound: 0.0102179
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152974, 0.0153212
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0028016, 0.0027917
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0109692, upper bound: 0.0109439
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0109598, upper bound: 0.0109598
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0153583, 0.0153379
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0028043, 0.0028128
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0103984, upper bound: 0.0103593
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0103854, upper bound: 0.0103841
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0153476, 0.0153682
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0028163, 0.0028083
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0098685, upper bound: 0.0098536
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0098726, upper bound: 0.0098523
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152450, 0.0152527
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027649, 0.0027617
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0099105, upper bound: 0.0098965
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0099105, upper bound: 0.0098965
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152553, 0.0152432
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027610, 0.0027660
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0106059, upper bound: 0.0105802
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0105851, upper bound: 0.0105968
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152486, 0.0152516
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027645, 0.0027632
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0105040, upper bound: 0.0105280
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0105225, upper bound: 0.0105053
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152574, 0.0152410
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027601, 0.0027669
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0107083, upper bound: 0.0107564
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0107512, upper bound: 0.0107199
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152473, 0.0152633
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027710, 0.0027638
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095231, upper bound: 0.0095114
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095205, upper bound: 0.0095115
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152427, 0.0152934
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027834, 0.0027619
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096829, upper bound: 0.0096716
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096829, upper bound: 0.0096715
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0150980, 0.0150724
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0026851, 0.0026953
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0108398, upper bound: 0.0108227
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0108516, upper bound: 0.0108170
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0150842, 0.0150857
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0026906, 0.0026895
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095205, upper bound: 0.0095115
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095205, upper bound: 0.0095115
time: 0.74 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.14 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0096142, upper bound: 0.0096335
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0096142, upper bound: 0.0096335
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0089434, upper bound: 0.0089398
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0089434, upper bound: 0.0089398
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0090379, upper bound: 0.0090199
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0090199, upper bound: 0.0090383
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0090379, upper bound: 0.0090199
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0090199, upper bound: 0.0090383
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0096142, upper bound: 0.0096335
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0096148, upper bound: 0.0096274
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0088870, upper bound: 0.0088872
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0088870, upper bound: 0.0088872
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0088872, upper bound: 0.0088870
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0088872, upper bound: 0.0088870
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0090946, upper bound: 0.0090820
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0090851, upper bound: 0.0090938
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0098972, upper bound: 0.0098935
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0098972, upper bound: 0.0098935
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0106743, upper bound: 0.0106736
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0106896, upper bound: 0.0106578
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0097892, upper bound: 0.0097700
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0097979, upper bound: 0.0097638
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0093132, upper bound: 0.0093040
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0093137, upper bound: 0.0093039
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0095698, upper bound: 0.0096057
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0095698, upper bound: 0.0096057
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0095708, upper bound: 0.0096051
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0095708, upper bound: 0.0096051
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0098217, upper bound: 0.0098123
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0098217, upper bound: 0.0098123
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0098217, upper bound: 0.0098123
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0098217, upper bound: 0.0098123
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0095465, upper bound: 0.0095355
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0095465, upper bound: 0.0095355
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0107696, upper bound: 0.0107853
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0107613, upper bound: 0.0108107
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0100754, upper bound: 0.0100786
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0100782, upper bound: 0.0100759
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0104109, upper bound: 0.0104126
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0104120, upper bound: 0.0104110
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0101748, upper bound: 0.0101775
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0101775, upper bound: 0.0101753
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0093146, upper bound: 0.0093175
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0093146, upper bound: 0.0093175
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0097019, upper bound: 0.0097029
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0097019, upper bound: 0.0097032
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0097019, upper bound: 0.0097029
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0097019, upper bound: 0.0097032
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0093726, upper bound: 0.0093765
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0093766, upper bound: 0.0093727
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0093726, upper bound: 0.0093765
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0093766, upper bound: 0.0093727
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0097990, upper bound: 0.0098068
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0098092, upper bound: 0.0097959
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0097830, upper bound: 0.0097959
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0097982, upper bound: 0.0097768
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0105442, upper bound: 0.0105564
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0105442, upper bound: 0.0105564
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0109697, upper bound: 0.0110234
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0109771, upper bound: 0.0110196
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0108156, upper bound: 0.0108353
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0108182, upper bound: 0.0108262
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0104305, upper bound: 0.0104506
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0104305, upper bound: 0.0104506
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0108203, upper bound: 0.0108119
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0108237, upper bound: 0.0108080
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0100790, upper bound: 0.0100697
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0100790, upper bound: 0.0100697
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0080903, upper bound: 0.0080991
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0080990, upper bound: 0.0080882
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0080938, upper bound: 0.0081115
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0081115, upper bound: 0.0080876
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0097478, upper bound: 0.0097490
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0097515, upper bound: 0.0097478
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0096574, upper bound: 0.0096524
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0096574, upper bound: 0.0096524
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0101369, upper bound: 0.0101172
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0101370, upper bound: 0.0101134
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0101369, upper bound: 0.0101172
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0101370, upper bound: 0.0101134
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0107996, upper bound: 0.0108590
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0107999, upper bound: 0.0108575
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0092813, upper bound: 0.0092973
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0092813, upper bound: 0.0092973
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0105524, upper bound: 0.0105673
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0105538, upper bound: 0.0105620
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0105524, upper bound: 0.0105673
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0105538, upper bound: 0.0105620
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0104161, upper bound: 0.0104056
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0103999, upper bound: 0.0104177
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0100421, upper bound: 0.0100728
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0100664, upper bound: 0.0100572
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0101000, upper bound: 0.0101137
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0101010, upper bound: 0.0101137
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0106567, upper bound: 0.0106636
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0106614, upper bound: 0.0106629
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0108035, upper bound: 0.0107923
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0108035, upper bound: 0.0107923
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0107884, upper bound: 0.0108086
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0107884, upper bound: 0.0108086
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0118941, upper bound: 0.0118626
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0118774, upper bound: 0.0118687
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0117867, upper bound: 0.0118511
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0117902, upper bound: 0.0118459
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0102275, upper bound: 0.0102179
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0102275, upper bound: 0.0102179
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0109692, upper bound: 0.0109439
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0109598, upper bound: 0.0109598
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0103984, upper bound: 0.0103593
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0103854, upper bound: 0.0103841
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0098685, upper bound: 0.0098536
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0098726, upper bound: 0.0098523
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0099105, upper bound: 0.0098965
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0099105, upper bound: 0.0098965
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0106059, upper bound: 0.0105802
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0105851, upper bound: 0.0105968
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0105040, upper bound: 0.0105280
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0105225, upper bound: 0.0105053
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0107083, upper bound: 0.0107564
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0107512, upper bound: 0.0107199
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0095231, upper bound: 0.0095114
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0095205, upper bound: 0.0095115
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0096829, upper bound: 0.0096716
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0096829, upper bound: 0.0096715
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0108398, upper bound: 0.0108227
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0108516, upper bound: 0.0108170
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0095205, upper bound: 0.0095115
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 1, lower bound: -0.0095205, upper bound: 0.0095115

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151277, 0.0151916
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027375, 0.0027110
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096106, upper bound: 0.0096335
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096142, upper bound: 0.0096255
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151357, 0.0151815
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027333, 0.0027143
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095971, upper bound: 0.0096224
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096030, upper bound: 0.0096104
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151215, 0.0151611
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027237, 0.0027072
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088132, upper bound: 0.0087880
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087942, upper bound: 0.0088102
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151278, 0.0151584
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027225, 0.0027098
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084500, upper bound: 0.0084486
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084500, upper bound: 0.0084486
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0150976, 0.0151143
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027083, 0.0027013
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0085252, upper bound: 0.0085106
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0085252, upper bound: 0.0085106
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151003, 0.0151119
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027073, 0.0027024
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087879, upper bound: 0.0088137
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087942, upper bound: 0.0088102
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0150999, 0.0151083
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027058, 0.0027023
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088585, upper bound: 0.0088379
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088530, upper bound: 0.0088401
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151054, 0.0151071
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027053, 0.0027046
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0085106, upper bound: 0.0085252
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0085106, upper bound: 0.0085252
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151301, 0.0151892
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027365, 0.0027120
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0094423, upper bound: 0.0094532
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0094373, upper bound: 0.0094611
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151475, 0.0151716
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027292, 0.0027192
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0094429, upper bound: 0.0094502
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0094385, upper bound: 0.0094555
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151704, 0.0151990
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027381, 0.0027262
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088829, upper bound: 0.0088872
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088870, upper bound: 0.0088829
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151742, 0.0151934
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027357, 0.0027278
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087103, upper bound: 0.0087064
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087048, upper bound: 0.0087103
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151697, 0.0151977
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027375, 0.0027259
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087432, upper bound: 0.0087396
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087399, upper bound: 0.0087432
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151782, 0.0151887
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027338, 0.0027295
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084500, upper bound: 0.0084486
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084500, upper bound: 0.0084486
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151272, 0.0151419
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027128, 0.0027067
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088137, upper bound: 0.0087879
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088137, upper bound: 0.0087879
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151331, 0.0151408
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027124, 0.0027092
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087942, upper bound: 0.0088093
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087942, upper bound: 0.0088093
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151831, 0.0152439
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027661, 0.0027402
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096754, upper bound: 0.0096761
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096787, upper bound: 0.0096729
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151854, 0.0152414
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027651, 0.0027411
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096621, upper bound: 0.0096707
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096746, upper bound: 0.0096615
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152022, 0.0152434
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027647, 0.0027469
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097926, upper bound: 0.0098112
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097926, upper bound: 0.0098112
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0152049, 0.0152414
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027639, 0.0027480
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0103476, upper bound: 0.0103191
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0103562, upper bound: 0.0103132
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151935, 0.0152242
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027569, 0.0027434
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095841, upper bound: 0.0095658
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095866, upper bound: 0.0095656
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151963, 0.0152218
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027558, 0.0027446
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096312, upper bound: 0.0095934
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096224, upper bound: 0.0095946
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151868, 0.0152095
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027521, 0.0027420
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091358, upper bound: 0.0091246
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091341, upper bound: 0.0091270
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151887, 0.0152073
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027513, 0.0027428
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093131, upper bound: 0.0093039
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093137, upper bound: 0.0093037
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151252, 0.0151742
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027263, 0.0027055
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090797, upper bound: 0.0091364
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090797, upper bound: 0.0091364
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151300, 0.0151694
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027243, 0.0027075
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0094026, upper bound: 0.0094307
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0094015, upper bound: 0.0094391
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151335, 0.0151650
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027225, 0.0027089
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090830, upper bound: 0.0091343
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090830, upper bound: 0.0091343
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151394, 0.0151609
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027208, 0.0027114
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090830, upper bound: 0.0091343
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090830, upper bound: 0.0091343
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151405, 0.0151446
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027253, 0.0027231
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093875, upper bound: 0.0094009
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093974, upper bound: 0.0093961
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151482, 0.0151340
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027208, 0.0027263
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093875, upper bound: 0.0094009
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093974, upper bound: 0.0093961
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0085975, 0.0075264, -0.0085975, 0.0075264, -0.0161239, 0.0161239
1: 0.9967313, 1.0140865, 0.9967313, 1.0140865, -0.0151024, 0.0151806
2: -0.0084901, 0.0077660, -0.0084901, 0.0077660, -0.0162561, 0.0162561
3: -0.0006925, 0.0026995, -0.0006925, 0.0026995, -0.0027402, 0.0027073
4: -0.0094619, 0.0022215, -0.0094619, 0.0022215, -0.0116834, 0.0116834
5: -0.0031360, 0.0114000, -0.0031360, 0.0114000, -0.0145360, 0.0145360
6: -0.0129813, 0.0022697, -0.0129813, 0.0022697, -0.0152510, 0.0152510
7: -0.0060811, 0.0014829, -0.0060811, 0.0014829, -0.0075640, 0.0075640
8: -0.0154618, -0.0008487, -0.0154618, -0.0008487, -0.0146131, 0.0146131
9: -0.0080999, 0.0087336, -0.0080999, 0.0087336, -0.0168335, 0.0168335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093875, upper bound: 0.0094009
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093974, upper bound: 0.0093961
time: 0.83 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 7.20 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0096106, upper bound: 0.0096335
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0096142, upper bound: 0.0096255
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0095971, upper bound: 0.0096224
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0096030, upper bound: 0.0096104
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0088132, upper bound: 0.0087880
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0087942, upper bound: 0.0088102
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0084500, upper bound: 0.0084486
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0084500, upper bound: 0.0084486
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0085252, upper bound: 0.0085106
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0085252, upper bound: 0.0085106
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0087879, upper bound: 0.0088137
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0087942, upper bound: 0.0088102
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0088585, upper bound: 0.0088379
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0088530, upper bound: 0.0088401
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0085106, upper bound: 0.0085252
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0085106, upper bound: 0.0085252
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0094423, upper bound: 0.0094532
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0094373, upper bound: 0.0094611
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0094429, upper bound: 0.0094502
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0094385, upper bound: 0.0094555
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0088829, upper bound: 0.0088872
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0088870, upper bound: 0.0088829
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0087103, upper bound: 0.0087064
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0087048, upper bound: 0.0087103
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0087432, upper bound: 0.0087396
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0087399, upper bound: 0.0087432
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0084500, upper bound: 0.0084486
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0084500, upper bound: 0.0084486
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0088137, upper bound: 0.0087879
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0088137, upper bound: 0.0087879
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0087942, upper bound: 0.0088093
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0087942, upper bound: 0.0088093
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0096754, upper bound: 0.0096761
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0096787, upper bound: 0.0096729
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0096621, upper bound: 0.0096707
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0096746, upper bound: 0.0096615
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0097926, upper bound: 0.0098112
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0097926, upper bound: 0.0098112
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0103476, upper bound: 0.0103191
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0103562, upper bound: 0.0103132
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0095841, upper bound: 0.0095658
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0095866, upper bound: 0.0095656
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0096312, upper bound: 0.0095934
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0096224, upper bound: 0.0095946
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0091358, upper bound: 0.0091246
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0091341, upper bound: 0.0091270
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0093131, upper bound: 0.0093039
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0093137, upper bound: 0.0093037
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0090797, upper bound: 0.0091364
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0090797, upper bound: 0.0091364
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0094026, upper bound: 0.0094307
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0094015, upper bound: 0.0094391
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0090830, upper bound: 0.0091343
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0090830, upper bound: 0.0091343
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0090830, upper bound: 0.0091343
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0090830, upper bound: 0.0091343
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0093875, upper bound: 0.0094009
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0093974, upper bound: 0.0093961
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0093875, upper bound: 0.0094009
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0093974, upper bound: 0.0093961
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0093875, upper bound: 0.0094009
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 1, lower bound: -0.0093974, upper bound: 0.0093961
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0098217, upper bound: 0.0098123
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0095465, upper bound: 0.0095355
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0095465, upper bound: 0.0095355
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0107696, upper bound: 0.0107853
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0107613, upper bound: 0.0108107
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0100754, upper bound: 0.0100786
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0100782, upper bound: 0.0100759
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0104109, upper bound: 0.0104126
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0104120, upper bound: 0.0104110
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0101748, upper bound: 0.0101775
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0101775, upper bound: 0.0101753
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0093146, upper bound: 0.0093175
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0093146, upper bound: 0.0093175
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0097019, upper bound: 0.0097029
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0097019, upper bound: 0.0097032
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0097019, upper bound: 0.0097029
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0097019, upper bound: 0.0097032
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0093726, upper bound: 0.0093765
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0093766, upper bound: 0.0093727
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0093726, upper bound: 0.0093765
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0093766, upper bound: 0.0093727
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0097990, upper bound: 0.0098068
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0098092, upper bound: 0.0097959
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0097830, upper bound: 0.0097959
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0097982, upper bound: 0.0097768
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0105442, upper bound: 0.0105564
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0105442, upper bound: 0.0105564
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0109697, upper bound: 0.0110234
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0109771, upper bound: 0.0110196
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0108156, upper bound: 0.0108353
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0108182, upper bound: 0.0108262
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0104305, upper bound: 0.0104506
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0104305, upper bound: 0.0104506
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0108203, upper bound: 0.0108119
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0108237, upper bound: 0.0108080
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0100790, upper bound: 0.0100697
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0100790, upper bound: 0.0100697
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0080903, upper bound: 0.0080991
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0080990, upper bound: 0.0080882
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0080938, upper bound: 0.0081115
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0081115, upper bound: 0.0080876
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0097478, upper bound: 0.0097490
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0097515, upper bound: 0.0097478
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0096574, upper bound: 0.0096524
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0096574, upper bound: 0.0096524
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0101369, upper bound: 0.0101172
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0101370, upper bound: 0.0101134
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0101369, upper bound: 0.0101172
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0101370, upper bound: 0.0101134
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0107996, upper bound: 0.0108590
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0107999, upper bound: 0.0108575
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0092813, upper bound: 0.0092973
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0092813, upper bound: 0.0092973
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0105524, upper bound: 0.0105673
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0105538, upper bound: 0.0105620
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0105524, upper bound: 0.0105673
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0105538, upper bound: 0.0105620
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0104161, upper bound: 0.0104056
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0103999, upper bound: 0.0104177
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0100421, upper bound: 0.0100728
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0100664, upper bound: 0.0100572
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0101000, upper bound: 0.0101137
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0101010, upper bound: 0.0101137
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0106567, upper bound: 0.0106636
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0106614, upper bound: 0.0106629
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0108035, upper bound: 0.0107923
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0108035, upper bound: 0.0107923
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0107884, upper bound: 0.0108086
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0107884, upper bound: 0.0108086
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0118941, upper bound: 0.0118626
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0118774, upper bound: 0.0118687
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0117867, upper bound: 0.0118511
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0117902, upper bound: 0.0118459
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0102275, upper bound: 0.0102179
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0102275, upper bound: 0.0102179
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0109692, upper bound: 0.0109439
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0109598, upper bound: 0.0109598
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0103984, upper bound: 0.0103593
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0103854, upper bound: 0.0103841
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0098685, upper bound: 0.0098536
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0098726, upper bound: 0.0098523
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0099105, upper bound: 0.0098965
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0099105, upper bound: 0.0098965
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0106059, upper bound: 0.0105802
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0105851, upper bound: 0.0105968
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0105040, upper bound: 0.0105280
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0105225, upper bound: 0.0105053
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0107083, upper bound: 0.0107564
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0107512, upper bound: 0.0107199
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0095231, upper bound: 0.0095114
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0095205, upper bound: 0.0095115
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0096829, upper bound: 0.0096716
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0096829, upper bound: 0.0096715
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0108398, upper bound: 0.0108227
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0108516, upper bound: 0.0108170
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0095205, upper bound: 0.0095115
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 1, lower bound: -0.0095205, upper bound: 0.0095115

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 3.35 + 597.84 = 601.19 seconds
