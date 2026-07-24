## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.001357455


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625)
1: (-0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128)
2: (0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554)
3: (-0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0126610, 0.0126610)
4: (-0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531)
5: (0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450)
6: (0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107)
7: (-0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458)
8: (0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171)
9: (0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0053181, 0.0053181)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.10 + 2.56 = 3.66 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0014289, upper bound: 0.0014289

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 172

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014216, upper bound: 0.0014147
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014147, upper bound: 0.0014215
time: 1.53 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.83 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.83
Output dim: 2, lower bound: -0.0014216, upper bound: 0.0014147
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.83
Output dim: 2, lower bound: -0.0014147, upper bound: 0.0014215

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0123235, 0.0121703
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052615, 0.0052784

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014199, upper bound: 0.0014129
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014198, upper bound: 0.0014129
time: 1.63 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0121703, 0.0123235
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052784, 0.0052615

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014101, upper bound: 0.0014169
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014100, upper bound: 0.0014169
time: 1.78 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.56 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.56
Output dim: 2, lower bound: -0.0014199, upper bound: 0.0014129
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.56
Output dim: 2, lower bound: -0.0014198, upper bound: 0.0014129
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.56
Output dim: 2, lower bound: -0.0014101, upper bound: 0.0014169
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.56
Output dim: 2, lower bound: -0.0014100, upper bound: 0.0014169

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0122259, 0.0120670
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052538, 0.0052712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 172

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014152, upper bound: 0.0014084
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014153, upper bound: 0.0014084
time: 1.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0122201, 0.0120740
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052545, 0.0052706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014177, upper bound: 0.0014110
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014178, upper bound: 0.0014103
time: 1.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0121090, 0.0122513
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052708, 0.0052552

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013948, upper bound: 0.0014125
time: 1.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014059, upper bound: 0.0014027
time: 1.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0120981, 0.0123235
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052784, 0.0052540

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 172

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013945, upper bound: 0.0014125
time: 2.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014059, upper bound: 0.0014028
time: 1.66 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.74 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.74
Output dim: 2, lower bound: -0.0014152, upper bound: 0.0014084
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.74
Output dim: 2, lower bound: -0.0014153, upper bound: 0.0014084
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.74
Output dim: 2, lower bound: -0.0014177, upper bound: 0.0014110
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.74
Output dim: 2, lower bound: -0.0014178, upper bound: 0.0014103
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.74
Output dim: 2, lower bound: -0.0013948, upper bound: 0.0014125
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.74
Output dim: 2, lower bound: -0.0014059, upper bound: 0.0014027
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.74
Output dim: 2, lower bound: -0.0013945, upper bound: 0.0014125
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.74
Output dim: 2, lower bound: -0.0014059, upper bound: 0.0014028

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0121692, 0.0119967
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052462, 0.0052651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 172

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013919, upper bound: 0.0014053
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014122, upper bound: 0.0013838
time: 1.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0121556, 0.0120670
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052538, 0.0052636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014010, upper bound: 0.0014042
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014108, upper bound: 0.0013930
time: 1.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0121741, 0.0120300
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052499, 0.0052658

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011222, upper bound: 0.0011203
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011222, upper bound: 0.0011203
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0121764, 0.0120280
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052497, 0.0052660

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 172

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013935, upper bound: 0.0014073
time: 1.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014149, upper bound: 0.0013868
time: 1.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0118116, 0.0120197
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052468, 0.0052240

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 172

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013748, upper bound: 0.0014095
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013917, upper bound: 0.0013882
time: 1.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0118850, 0.0119539
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052396, 0.0052320

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014042, upper bound: 0.0014008
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014041, upper bound: 0.0014011
time: 1.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0118008, 0.0120929
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052543, 0.0052228

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 172

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013746, upper bound: 0.0014095
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013914, upper bound: 0.0013884
time: 1.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0118793, 0.0120270
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052471, 0.0052314

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014030, upper bound: 0.0014007
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014038, upper bound: 0.0014008
time: 1.69 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.28 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 2, lower bound: -0.0013919, upper bound: 0.0014053
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 2, lower bound: -0.0014122, upper bound: 0.0013838
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 2, lower bound: -0.0014010, upper bound: 0.0014042
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 2, lower bound: -0.0014108, upper bound: 0.0013930
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.28
Output dim: 2, lower bound: -0.0011222, upper bound: 0.0011203
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.28
Output dim: 2, lower bound: -0.0011222, upper bound: 0.0011203
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 2, lower bound: -0.0013935, upper bound: 0.0014073
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 2, lower bound: -0.0014149, upper bound: 0.0013868
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 2, lower bound: -0.0013748, upper bound: 0.0014095
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 2, lower bound: -0.0013917, upper bound: 0.0013882
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 2, lower bound: -0.0014042, upper bound: 0.0014008
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 2, lower bound: -0.0014041, upper bound: 0.0014011
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 2, lower bound: -0.0013746, upper bound: 0.0014095
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 2, lower bound: -0.0013914, upper bound: 0.0013884
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 2, lower bound: -0.0014030, upper bound: 0.0014007
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 2, lower bound: -0.0014038, upper bound: 0.0014008

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0114261, 0.0114971
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051975, 0.0051897

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013817, upper bound: 0.0014011
time: 1.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013869, upper bound: 0.0013897
time: 1.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0116384, 0.0112536
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051708, 0.0052131

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014100, upper bound: 0.0013818
time: 1.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014101, upper bound: 0.0013815
time: 1.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0118494, 0.0118399
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052306, 0.0052319

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 172

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013813, upper bound: 0.0014012
time: 1.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013979, upper bound: 0.0013793
time: 1.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0119147, 0.0117613
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052220, 0.0052391

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014087, upper bound: 0.0013909
time: 1.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014088, upper bound: 0.0013907
time: 1.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0114311, 0.0115181
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052004, 0.0051908

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013891, upper bound: 0.0014026
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013889, upper bound: 0.0014027
time: 1.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0116616, 0.0112828
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051745, 0.0052161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014100, upper bound: 0.0013822
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014100, upper bound: 0.0013823
time: 1.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110710, 0.0115200
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051986, 0.0051493

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013727, upper bound: 0.0014075
time: 1.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013727, upper bound: 0.0014073
time: 1.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0113075, 0.0112791
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051721, 0.0051753

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013896, upper bound: 0.0013862
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013897, upper bound: 0.0013862
time: 1.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0117794, 0.0118437
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052313, 0.0052242

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 172

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013798, upper bound: 0.0013978
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014012, upper bound: 0.0013807
time: 1.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0117748, 0.0118494
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052319, 0.0052237

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 172

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014013, upper bound: 0.0013989
time: 1.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014022, upper bound: 0.0013990
time: 1.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110601, 0.0115896
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052062, 0.0051481

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013725, upper bound: 0.0014075
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013725, upper bound: 0.0014073
time: 2.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0113036, 0.0113486
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051797, 0.0051748

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013893, upper bound: 0.0013864
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013894, upper bound: 0.0013864
time: 1.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0118334, 0.0119840
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052425, 0.0052265

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014014, upper bound: 0.0013987
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014013, upper bound: 0.0013991
time: 1.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0118394, 0.0119818
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052423, 0.0052272

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 172

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014022, upper bound: 0.0013987
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014021, upper bound: 0.0013991
time: 1.54 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 6.30 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.30
Output dim: 2, lower bound: -0.0013817, upper bound: 0.0014011
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.30
Output dim: 2, lower bound: -0.0013869, upper bound: 0.0013897
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.30
Output dim: 2, lower bound: -0.0014100, upper bound: 0.0013818
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.30
Output dim: 2, lower bound: -0.0014101, upper bound: 0.0013815
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.30
Output dim: 2, lower bound: -0.0013813, upper bound: 0.0014012
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.30
Output dim: 2, lower bound: -0.0013979, upper bound: 0.0013793
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.30
Output dim: 2, lower bound: -0.0014087, upper bound: 0.0013909
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.30
Output dim: 2, lower bound: -0.0014088, upper bound: 0.0013907
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.30
Output dim: 2, lower bound: -0.0013891, upper bound: 0.0014026
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.30
Output dim: 2, lower bound: -0.0013889, upper bound: 0.0014027
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.30
Output dim: 2, lower bound: -0.0014100, upper bound: 0.0013822
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.30
Output dim: 2, lower bound: -0.0014100, upper bound: 0.0013823
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.30
Output dim: 2, lower bound: -0.0013727, upper bound: 0.0014075
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.30
Output dim: 2, lower bound: -0.0013727, upper bound: 0.0014073
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.30
Output dim: 2, lower bound: -0.0013896, upper bound: 0.0013862
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.30
Output dim: 2, lower bound: -0.0013897, upper bound: 0.0013862
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.30
Output dim: 2, lower bound: -0.0013798, upper bound: 0.0013978
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.30
Output dim: 2, lower bound: -0.0014012, upper bound: 0.0013807
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.30
Output dim: 2, lower bound: -0.0014013, upper bound: 0.0013989
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.30
Output dim: 2, lower bound: -0.0014022, upper bound: 0.0013990
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.30
Output dim: 2, lower bound: -0.0013725, upper bound: 0.0014075
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.30
Output dim: 2, lower bound: -0.0013725, upper bound: 0.0014073
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.30
Output dim: 2, lower bound: -0.0013893, upper bound: 0.0013864
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.30
Output dim: 2, lower bound: -0.0013894, upper bound: 0.0013864
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.30
Output dim: 2, lower bound: -0.0014014, upper bound: 0.0013987
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.30
Output dim: 2, lower bound: -0.0014013, upper bound: 0.0013991
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.30
Output dim: 2, lower bound: -0.0014022, upper bound: 0.0013987
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.30
Output dim: 2, lower bound: -0.0014021, upper bound: 0.0013991

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111244, 0.0112797
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051753, 0.0051583

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013795, upper bound: 0.0013992
time: 1.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013795, upper bound: 0.0013983
time: 1.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111863, 0.0111955
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051661, 0.0051651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013848, upper bound: 0.0013876
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013848, upper bound: 0.0013875
time: 1.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0115913, 0.0112108
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051667, 0.0052085

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013961, upper bound: 0.0013771
time: 1.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014056, upper bound: 0.0013699
time: 1.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0115940, 0.0112065
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051663, 0.0052088

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013960, upper bound: 0.0013769
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014058, upper bound: 0.0013698
time: 1.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111108, 0.0113482
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051828, 0.0051568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013792, upper bound: 0.0013992
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0013984
time: 1.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0113339, 0.0110991
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051554, 0.0051813

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013959, upper bound: 0.0013772
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013959, upper bound: 0.0013770
time: 1.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0118706, 0.0117217
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052177, 0.0052343

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 172

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013846, upper bound: 0.0013879
time: 1.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014057, upper bound: 0.0013700
time: 1.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0118716, 0.0117175
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052172, 0.0052344

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 172

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013845, upper bound: 0.0013877
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014058, upper bound: 0.0013699
time: 1.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0113742, 0.0114489
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051929, 0.0051847

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013788, upper bound: 0.0013984
time: 1.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013841, upper bound: 0.0013877
time: 1.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0113619, 0.0115181
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052004, 0.0051833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013784, upper bound: 0.0013985
time: 1.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013839, upper bound: 0.0013881
time: 1.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0115926, 0.0112136
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051670, 0.0052086

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013957, upper bound: 0.0013774
time: 2.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014058, upper bound: 0.0013709
time: 1.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0115924, 0.0112828
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051745, 0.0052086

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013955, upper bound: 0.0013776
time: 1.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014058, upper bound: 0.0013711
time: 1.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110248, 0.0114800
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051947, 0.0051447

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013711, upper bound: 0.0014058
time: 2.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0014059
time: 1.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110308, 0.0114739
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051940, 0.0051453

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013711, upper bound: 0.0014056
time: 2.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0014056
time: 1.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112613, 0.0112370
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051680, 0.0051706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013880, upper bound: 0.0013839
time: 2.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013877, upper bound: 0.0013846
time: 1.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112686, 0.0112329
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051675, 0.0051714

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013881, upper bound: 0.0013839
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013879, upper bound: 0.0013846
time: 2.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110408, 0.0113340
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051813, 0.0051491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013776, upper bound: 0.0013956
time: 1.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013777, upper bound: 0.0013956
time: 1.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112792, 0.0111051
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051562, 0.0051753

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013985, upper bound: 0.0013783
time: 1.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013992, upper bound: 0.0013786
time: 1.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0117307, 0.0118054
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052271, 0.0052189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 172

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 172

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013770, upper bound: 0.0013959
time: 1.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013984, upper bound: 0.0013791
time: 1.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0117373, 0.0118053
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052271, 0.0052197

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 172

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 172

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013772, upper bound: 0.0013959
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013991, upper bound: 0.0013792
time: 1.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110139, 0.0115497
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052022, 0.0051435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0014058
time: 1.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013698, upper bound: 0.0014058
time: 1.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110182, 0.0115436
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052016, 0.0051439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013709, upper bound: 0.0014056
time: 1.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0014057
time: 1.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112574, 0.0113067
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051755, 0.0051702

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013877, upper bound: 0.0013841
time: 2.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013875, upper bound: 0.0013848
time: 2.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112658, 0.0113026
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051751, 0.0051711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013877, upper bound: 0.0013841
time: 1.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013876, upper bound: 0.0013848
time: 1.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0117296, 0.0118728
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052342, 0.0052188

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 172

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013774, upper bound: 0.0013957
time: 1.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013984, upper bound: 0.0013788
time: 2.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0117251, 0.0118764
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052346, 0.0052183

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 172

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0013961
time: 1.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013983, upper bound: 0.0013796
time: 1.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0117338, 0.0118706
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052340, 0.0052193

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 172

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 172

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013775, upper bound: 0.0013957
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013992, upper bound: 0.0013789
time: 1.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0117311, 0.0118763
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052346, 0.0052190

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 172

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 172

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013770, upper bound: 0.0013961
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013992, upper bound: 0.0013796
time: 1.84 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 6.69 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013795, upper bound: 0.0013992
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013795, upper bound: 0.0013983
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013848, upper bound: 0.0013876
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013848, upper bound: 0.0013875
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013961, upper bound: 0.0013771
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0014056, upper bound: 0.0013699
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013960, upper bound: 0.0013769
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0014058, upper bound: 0.0013698
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013792, upper bound: 0.0013992
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0013984
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013959, upper bound: 0.0013772
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013959, upper bound: 0.0013770
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013846, upper bound: 0.0013879
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0014057, upper bound: 0.0013700
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013845, upper bound: 0.0013877
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0014058, upper bound: 0.0013699
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013788, upper bound: 0.0013984
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013841, upper bound: 0.0013877
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013784, upper bound: 0.0013985
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013839, upper bound: 0.0013881
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013957, upper bound: 0.0013774
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0014058, upper bound: 0.0013709
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013955, upper bound: 0.0013776
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0014058, upper bound: 0.0013711
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013711, upper bound: 0.0014058
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0014059
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013711, upper bound: 0.0014056
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0014056
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013880, upper bound: 0.0013839
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013877, upper bound: 0.0013846
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013881, upper bound: 0.0013839
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013879, upper bound: 0.0013846
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013776, upper bound: 0.0013956
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013777, upper bound: 0.0013956
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013985, upper bound: 0.0013783
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013992, upper bound: 0.0013786
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013770, upper bound: 0.0013959
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013984, upper bound: 0.0013791
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013772, upper bound: 0.0013959
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013991, upper bound: 0.0013792
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0014058
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013698, upper bound: 0.0014058
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013709, upper bound: 0.0014056
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0014057
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013877, upper bound: 0.0013841
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013875, upper bound: 0.0013848
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013877, upper bound: 0.0013841
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013876, upper bound: 0.0013848
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013774, upper bound: 0.0013957
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013984, upper bound: 0.0013788
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0013961
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013983, upper bound: 0.0013796
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013775, upper bound: 0.0013957
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013992, upper bound: 0.0013789
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013770, upper bound: 0.0013961
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 2, lower bound: -0.0013992, upper bound: 0.0013796

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110792, 0.0112433
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051718, 0.0051538

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 111

Time for candidate selection: 2.07 seconds

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013542, upper bound: 0.0013934
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013726, upper bound: 0.0013600
time: 1.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110784, 0.0112345
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051709, 0.0051537

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 13

Time for candidate selection: 2.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013543, upper bound: 0.0013879
time: 1.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013692, upper bound: 0.0013657
time: 2.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111411, 0.0111586
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051625, 0.0051606

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 206

Time for candidate selection: 2.07 seconds

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013840, upper bound: 0.0013665
time: 1.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013793, upper bound: 0.0013869
time: 1.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111427, 0.0111502
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051616, 0.0051608

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 187

Time for candidate selection: 2.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 248

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010657, upper bound: 0.0010672
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010657, upper bound: 0.0010672
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112915, 0.0109914
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051442, 0.0051771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 2.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 142

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013633, upper bound: 0.0013706
time: 2.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013900, upper bound: 0.0013495
time: 2.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0113675, 0.0109110
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051354, 0.0051855

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 141

Time for candidate selection: 2.06 seconds

### Candidate
type: RSZ, layer: 3, pos: 206

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013678, upper bound: 0.0013338
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013678, upper bound: 0.0013338
time: 1.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112942, 0.0109853
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051435, 0.0051774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 142

Time for candidate selection: 2.06 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012502, upper bound: 0.0012387
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012502, upper bound: 0.0012387
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0113713, 0.0109067
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051349, 0.0051859

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 65

Time for candidate selection: 2.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 248

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010718, upper bound: 0.0010622
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010718, upper bound: 0.0010622
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110656, 0.0113123
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051793, 0.0051523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 13

Time for candidate selection: 2.02 seconds

### Candidate
type: RSZ, layer: 3, pos: 142

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013724, upper bound: 0.0013919
time: 2.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013724, upper bound: 0.0013917
time: 2.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110657, 0.0113035
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051784, 0.0051523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 2.05 seconds

### Candidate
type: RSZ, layer: 3, pos: 206

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013396, upper bound: 0.0013630
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013396, upper bound: 0.0013631
time: 1.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112887, 0.0110604
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051517, 0.0051768

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 2.08 seconds

### Candidate
type: RSZ, layer: 3, pos: 142

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 206

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013567, upper bound: 0.0013416
time: 1.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013566, upper bound: 0.0013416
time: 1.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112922, 0.0110544
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051510, 0.0051772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 142

Time for candidate selection: 2.07 seconds

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013626, upper bound: 0.0013704
time: 1.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013898, upper bound: 0.0013499
time: 1.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111309, 0.0112276
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051700, 0.0051595

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 206

Time for candidate selection: 2.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013838, upper bound: 0.0013665
time: 2.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013793, upper bound: 0.0013871
time: 2.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0113661, 0.0109800
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051429, 0.0051853

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 2.06 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013648, upper bound: 0.0013601
time: 2.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013949, upper bound: 0.0013518
time: 1.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111319, 0.0112193
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051691, 0.0051596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 248

Time for candidate selection: 2.07 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013778, upper bound: 0.0013732
time: 1.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013751, upper bound: 0.0013809
time: 2.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0113696, 0.0109758
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051424, 0.0051857

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 141

Time for candidate selection: 2.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013989, upper bound: 0.0013570
time: 1.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013925, upper bound: 0.0013631
time: 1.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110744, 0.0112306
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051705, 0.0051533

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 13

Time for candidate selection: 2.07 seconds

### Candidate
type: RSZ, layer: 3, pos: 206

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013395, upper bound: 0.0013631
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013395, upper bound: 0.0013631
time: 1.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111403, 0.0111491
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051615, 0.0051605

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 141

Time for candidate selection: 2.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 142

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013833, upper bound: 0.0013656
time: 1.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013794, upper bound: 0.0013869
time: 1.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110621, 0.0112996
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051780, 0.0051520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 13

Time for candidate selection: 2.09 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013557, upper bound: 0.0013880
time: 1.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013682, upper bound: 0.0013621
time: 2.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111298, 0.0112181
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051690, 0.0051594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 2.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013604, upper bound: 0.0013777
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013738, upper bound: 0.0013589
time: 2.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112928, 0.0109899
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051440, 0.0051773

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 13

Time for candidate selection: 2.07 seconds

### Candidate
type: RSZ, layer: 3, pos: 248

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010701, upper bound: 0.0010637
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010701, upper bound: 0.0010637
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0113719, 0.0109138
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051357, 0.0051860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 2.06 seconds

### Candidate
type: RSZ, layer: 3, pos: 248

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010722, upper bound: 0.0010622
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010722, upper bound: 0.0010622
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112926, 0.0110589
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051515, 0.0051773

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 2.08 seconds

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013948, upper bound: 0.0013641
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013847, upper bound: 0.0013768
time: 1.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0113728, 0.0109828
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051432, 0.0051861

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 13

Time for candidate selection: 2.05 seconds

### Candidate
type: RSZ, layer: 3, pos: 142

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013693, upper bound: 0.0013611
time: 1.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013951, upper bound: 0.0013509
time: 1.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0109237, 0.0113728
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051861, 0.0051368

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 13

Time for candidate selection: 2.08 seconds

### Candidate
type: RSZ, layer: 3, pos: 248

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010622, upper bound: 0.0010722
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010622, upper bound: 0.0010722
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0109176, 0.0113696
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051857, 0.0051361

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 206

Time for candidate selection: 2.08 seconds

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013690, upper bound: 0.0013904
time: 1.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013588, upper bound: 0.0014051
time: 2.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0109260, 0.0113667
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051854, 0.0051370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 111

Time for candidate selection: 2.07 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012327, upper bound: 0.0012586
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012327, upper bound: 0.0012586
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0109236, 0.0113661
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051853, 0.0051367

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 142

Time for candidate selection: 2.08 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013642, upper bound: 0.0013973
time: 1.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013642, upper bound: 0.0013969
time: 2.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111532, 0.0111298
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051594, 0.0051620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 187

Time for candidate selection: 2.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013814, upper bound: 0.0013741
time: 1.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013747, upper bound: 0.0013771
time: 1.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111541, 0.0111319
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051596, 0.0051620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 142

Time for candidate selection: 2.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013695, upper bound: 0.0013778
time: 2.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013652, upper bound: 0.0013841
time: 1.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111596, 0.0111257
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051589, 0.0051627

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 142

Time for candidate selection: 2.05 seconds

### Candidate
type: RSZ, layer: 3, pos: 206

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013519, upper bound: 0.0013473
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013519, upper bound: 0.0013473
time: 1.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111614, 0.0111309
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051595, 0.0051629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 248

Time for candidate selection: 2.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012495, upper bound: 0.0012406
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012495, upper bound: 0.0012406
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0109955, 0.0112926
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051773, 0.0051446

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 141

Time for candidate selection: 2.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012387, upper bound: 0.0012504
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012387, upper bound: 0.0012504
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110019, 0.0112887
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051768, 0.0051453

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 13

Time for candidate selection: 2.01 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013728, upper bound: 0.0013845
time: 1.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013728, upper bound: 0.0013825
time: 2.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112340, 0.0110621
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051520, 0.0051708

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 187

Time for candidate selection: 2.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013980, upper bound: 0.0013719
time: 1.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013848, upper bound: 0.0013779
time: 1.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112416, 0.0110599
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051517, 0.0051717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 141

Time for candidate selection: 2.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013988, upper bound: 0.0013719
time: 1.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013875, upper bound: 0.0013781
time: 1.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0109910, 0.0112922
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051772, 0.0051441

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187

Time for candidate selection: 2.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 206

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013412, upper bound: 0.0013576
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013412, upper bound: 0.0013576
time: 1.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112361, 0.0110657
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051523, 0.0051711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 248

Time for candidate selection: 2.07 seconds

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013596, upper bound: 0.0013723
time: 2.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013926, upper bound: 0.0013537
time: 2.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0109976, 0.0112887
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051768, 0.0051449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 65

Time for candidate selection: 2.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 142

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013693, upper bound: 0.0013888
time: 1.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013580, upper bound: 0.0013901
time: 2.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112436, 0.0110656
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051523, 0.0051719

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 248

Time for candidate selection: 2.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013601, upper bound: 0.0013724
time: 1.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013934, upper bound: 0.0013537
time: 1.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0109138, 0.0114419
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051936, 0.0051357

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 2.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013643, upper bound: 0.0013918
time: 2.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013591, upper bound: 0.0013988
time: 1.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0109067, 0.0114387
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051932, 0.0051349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 187

Time for candidate selection: 2.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013642, upper bound: 0.0013973
time: 1.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013643, upper bound: 0.0013970
time: 2.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0109158, 0.0114357
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051929, 0.0051359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 2.08 seconds

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013701, upper bound: 0.0013896
time: 1.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013589, upper bound: 0.0014048
time: 1.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0109110, 0.0114352
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051928, 0.0051354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 3.66 + 596.80 = 600.46 seconds
