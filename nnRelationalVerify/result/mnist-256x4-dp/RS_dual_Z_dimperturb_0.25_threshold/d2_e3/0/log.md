## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00886005


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0012166, 0.0012166)
1: (-0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0030593, 0.0030593)
2: (0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0043976, 0.0043976)
3: (-0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0032380, 0.0032380)
4: (-0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0037253, 0.0037253)
5: (0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0032260, 0.0032260)
6: (0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432)
7: (-0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0065400, 0.0065400)
8: (0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0207084, 0.0207084)
9: (0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0056235, 0.0056235)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.40 + 1.57 = 2.96 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0132417, upper bound: 0.0132417

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0103433, upper bound: 0.0103433
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0103433, upper bound: 0.0103433
time: 0.53 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.23 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.23
Output dim: 8, lower bound: -0.0103433, upper bound: 0.0103433
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.23
Output dim: 8, lower bound: -0.0103433, upper bound: 0.0103433

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0012169, 0.0012162
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0030606, 0.0030576
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0043952, 0.0043997
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0032362, 0.0032396
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0037236, 0.0037267
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0032241, 0.0032275
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0065433, 0.0065360
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0207178, 0.0206969
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0056201, 0.0056262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102638, upper bound: 0.0102844
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102844, upper bound: 0.0102638
time: 0.52 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0012162, 0.0012166
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0030576, 0.0030593
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0043976, 0.0043952
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0032380, 0.0032362
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0037253, 0.0037236
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0032260, 0.0032241
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0065360, 0.0065400
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0206969, 0.0207084
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0056235, 0.0056201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102638, upper bound: 0.0102844
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102844, upper bound: 0.0102638
time: 0.51 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.47 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.47
Output dim: 8, lower bound: -0.0102638, upper bound: 0.0102844
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.47
Output dim: 8, lower bound: -0.0102844, upper bound: 0.0102638
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.47
Output dim: 8, lower bound: -0.0102638, upper bound: 0.0102844
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.47
Output dim: 8, lower bound: -0.0102844, upper bound: 0.0102638

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0012125, 0.0012123
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0030361, 0.0030350
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0043592, 0.0043608
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0032095, 0.0032107
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0037040, 0.0037051
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0031975, 0.0031987
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0064820, 0.0064793
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0205412, 0.0205335
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0055708, 0.0055731

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102231, upper bound: 0.0102388
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102165, upper bound: 0.0102495
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0012127, 0.0012119
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0030369, 0.0030331
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0043563, 0.0043620
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0032074, 0.0032116
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0037020, 0.0037059
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0031954, 0.0031996
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0064840, 0.0064747
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0205468, 0.0205203
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0055670, 0.0055747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102495, upper bound: 0.0102165
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102388, upper bound: 0.0102231
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0012119, 0.0012127
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0030331, 0.0030367
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0043618, 0.0043563
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0032114, 0.0032074
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0037058, 0.0037020
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0031994, 0.0031954
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0064747, 0.0064836
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0205203, 0.0205457
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0055744, 0.0055670

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102231, upper bound: 0.0102388
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102165, upper bound: 0.0102495
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0012123, 0.0012123
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0030350, 0.0030348
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0043590, 0.0043592
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0032093, 0.0032095
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0037038, 0.0037040
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0031973, 0.0031975
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0064793, 0.0064790
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0205335, 0.0205325
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0055705, 0.0055708

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102495, upper bound: 0.0102165
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102388, upper bound: 0.0102231
time: 0.54 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.46 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 8, lower bound: -0.0102231, upper bound: 0.0102388
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 8, lower bound: -0.0102165, upper bound: 0.0102495
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 8, lower bound: -0.0102495, upper bound: 0.0102165
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 8, lower bound: -0.0102388, upper bound: 0.0102231
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 8, lower bound: -0.0102231, upper bound: 0.0102388
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 8, lower bound: -0.0102165, upper bound: 0.0102495
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 8, lower bound: -0.0102495, upper bound: 0.0102165
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 8, lower bound: -0.0102388, upper bound: 0.0102231

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0012090, 0.0012084
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0030245, 0.0030216
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0043394, 0.0043436
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0031938, 0.0031970
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0036910, 0.0036939
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0031817, 0.0031849
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0064560, 0.0064491
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0204613, 0.0204416
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0055462, 0.0055520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101706, upper bound: 0.0102269
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102111, upper bound: 0.0101815
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0012086, 0.0012088
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0030227, 0.0030237
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0043424, 0.0043410
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0031961, 0.0031951
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0036931, 0.0036921
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0031840, 0.0031830
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0064517, 0.0064540
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0204492, 0.0204557
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0055504, 0.0055485

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101645, upper bound: 0.0102376
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102045, upper bound: 0.0101894
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0012093, 0.0012080
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0030258, 0.0030198
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0043366, 0.0043456
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0031917, 0.0031985
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0036891, 0.0036953
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0031796, 0.0031864
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0064591, 0.0064445
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0204704, 0.0204284
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0055423, 0.0055547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101894, upper bound: 0.0102045
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102376, upper bound: 0.0101645
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0012088, 0.0012085
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0030235, 0.0030221
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0043401, 0.0043422
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0031944, 0.0031960
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0036915, 0.0036930
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0031822, 0.0031839
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0064537, 0.0064502
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0204549, 0.0204448
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0055472, 0.0055501

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101815, upper bound: 0.0102111
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102269, upper bound: 0.0101706
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0012085, 0.0012088
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0030221, 0.0030234
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0043419, 0.0043401
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0031958, 0.0031944
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0036928, 0.0036915
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0031836, 0.0031822
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0064502, 0.0064532
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0204448, 0.0204535
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0055497, 0.0055472

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101706, upper bound: 0.0102269
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102111, upper bound: 0.0101815
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0012080, 0.0012092
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0030198, 0.0030254
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0043450, 0.0043366
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0031980, 0.0031917
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0036949, 0.0036891
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0031859, 0.0031796
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0064445, 0.0064582
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0204284, 0.0204677
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0055539, 0.0055423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101645, upper bound: 0.0102376
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102045, upper bound: 0.0101894
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0012088, 0.0012084
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0030237, 0.0030215
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0043391, 0.0043424
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0031936, 0.0031961
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0036908, 0.0036931
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0031815, 0.0031840
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0064540, 0.0064486
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0204557, 0.0204404
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0055459, 0.0055504

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101894, upper bound: 0.0102045
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102376, upper bound: 0.0101645
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0012084, 0.0012089
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0030216, 0.0030238
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0043426, 0.0043394
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0031963, 0.0031938
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0036933, 0.0036910
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0031842, 0.0031817
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0064491, 0.0064543
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0204416, 0.0204568
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0055507, 0.0055462

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101815, upper bound: 0.0102111
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102269, upper bound: 0.0101706
time: 0.54 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.39 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 8, lower bound: -0.0101706, upper bound: 0.0102269
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 8, lower bound: -0.0102111, upper bound: 0.0101815
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 8, lower bound: -0.0101645, upper bound: 0.0102376
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 8, lower bound: -0.0102045, upper bound: 0.0101894
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 8, lower bound: -0.0101894, upper bound: 0.0102045
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 8, lower bound: -0.0102376, upper bound: 0.0101645
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 8, lower bound: -0.0101815, upper bound: 0.0102111
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 8, lower bound: -0.0102269, upper bound: 0.0101706
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 8, lower bound: -0.0101706, upper bound: 0.0102269
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 8, lower bound: -0.0102111, upper bound: 0.0101815
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 8, lower bound: -0.0101645, upper bound: 0.0102376
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 8, lower bound: -0.0102045, upper bound: 0.0101894
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 8, lower bound: -0.0101894, upper bound: 0.0102045
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 8, lower bound: -0.0102376, upper bound: 0.0101645
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 8, lower bound: -0.0101815, upper bound: 0.0102111
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 8, lower bound: -0.0102269, upper bound: 0.0101706

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011885, 0.0011910
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0029239, 0.0029357
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0041930, 0.0041754
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0030763, 0.0030631
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0036427, 0.0036305
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0030638, 0.0030506
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0061642, 0.0061928
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0196860, 0.0197681
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0053142, 0.0052901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089636, upper bound: 0.0090754
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090455, upper bound: 0.0090027
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011912, 0.0011879
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0029366, 0.0029211
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0041711, 0.0041944
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0030599, 0.0030774
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0036276, 0.0036437
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0030474, 0.0030648
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0061951, 0.0061573
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0197747, 0.0196663
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0052843, 0.0053161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090048, upper bound: 0.0090394
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090757, upper bound: 0.0089587
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011881, 0.0011912
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0029222, 0.0029365
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0041942, 0.0041728
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0030773, 0.0030612
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0036436, 0.0036287
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0030647, 0.0030486
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0061600, 0.0061949
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0196739, 0.0197740
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0053159, 0.0052865

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089532, upper bound: 0.0090861
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090308, upper bound: 0.0090084
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011910, 0.0011883
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0029358, 0.0029231
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0041742, 0.0041932
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0030622, 0.0030765
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0036297, 0.0036429
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0030497, 0.0030639
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0061932, 0.0061622
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0197692, 0.0196804
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0052884, 0.0053145

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089960, upper bound: 0.0090559
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090593, upper bound: 0.0089681
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011888, 0.0011904
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0029252, 0.0029330
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0041889, 0.0041773
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0030733, 0.0030646
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0036399, 0.0036319
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0030607, 0.0030520
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0061673, 0.0061863
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0196951, 0.0197493
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0053086, 0.0052927

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089681, upper bound: 0.0090593
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090559, upper bound: 0.0089960
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011918, 0.0011875
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0029393, 0.0029192
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0041683, 0.0041983
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0030578, 0.0030803
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0036256, 0.0036464
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0030453, 0.0030678
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0062016, 0.0061527
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0197931, 0.0196531
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0052804, 0.0053215

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090084, upper bound: 0.0090308
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090861, upper bound: 0.0089532
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011883, 0.0011906
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0029230, 0.0029339
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0041903, 0.0041740
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0030743, 0.0030621
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0036409, 0.0036296
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0030618, 0.0030495
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0061619, 0.0061885
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0196795, 0.0197557
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0053105, 0.0052882

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089587, upper bound: 0.0090757
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090394, upper bound: 0.0090048
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011916, 0.0011880
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0029386, 0.0029216
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0041718, 0.0041973
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0030604, 0.0030796
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0036281, 0.0036457
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0030479, 0.0030670
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0061999, 0.0061584
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0197883, 0.0196695
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0052852, 0.0053201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090027, upper bound: 0.0090455
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090754, upper bound: 0.0089636
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011880, 0.0011914
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0029216, 0.0029373
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0041955, 0.0041718
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0030782, 0.0030604
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0036445, 0.0036281
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0030656, 0.0030479
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0061584, 0.0061969
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0196695, 0.0197797
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0053176, 0.0052852

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089636, upper bound: 0.0090754
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090455, upper bound: 0.0090027
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011906, 0.0011883
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0029339, 0.0029228
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0041736, 0.0041903
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0030618, 0.0030743
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0036293, 0.0036409
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0030493, 0.0030618
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0061885, 0.0061614
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0197557, 0.0196779
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0052877, 0.0053105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090048, upper bound: 0.0090394
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090757, upper bound: 0.0089587
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011875, 0.0011915
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0029192, 0.0029382
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0041967, 0.0041683
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0030792, 0.0030578
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0036454, 0.0036256
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0030666, 0.0030453
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0061527, 0.0061990
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0196531, 0.0197857
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0053194, 0.0052804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089532, upper bound: 0.0090861
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090308, upper bound: 0.0090084
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011904, 0.0011887
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0029330, 0.0029248
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0041767, 0.0041889
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0030641, 0.0030733
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0036314, 0.0036399
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0030515, 0.0030607
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0061863, 0.0061663
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0197493, 0.0196921
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0052918, 0.0053086

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089960, upper bound: 0.0090559
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090593, upper bound: 0.0089681
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011883, 0.0011908
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0029231, 0.0029347
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0041914, 0.0041742
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0030752, 0.0030622
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0036417, 0.0036297
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0030626, 0.0030497
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0061622, 0.0061903
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0196804, 0.0197609
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0053121, 0.0052884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089681, upper bound: 0.0090593
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090559, upper bound: 0.0089960
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011912, 0.0011879
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0029365, 0.0029209
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0041708, 0.0041942
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0030597, 0.0030773
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0036274, 0.0036436
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0030471, 0.0030647
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0061949, 0.0061568
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0197740, 0.0196648
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0052838, 0.0053159

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090084, upper bound: 0.0090308
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090861, upper bound: 0.0089532
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011879, 0.0011910
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0029211, 0.0029356
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0041928, 0.0041711
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0030762, 0.0030599
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0036426, 0.0036276
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0030636, 0.0030474
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0061573, 0.0061926
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0196663, 0.0197674
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0053140, 0.0052843

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089587, upper bound: 0.0090757
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090394, upper bound: 0.0090048
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011910, 0.0011884
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0029357, 0.0029232
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0041743, 0.0041930
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0030623, 0.0030763
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0036298, 0.0036427
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0030498, 0.0030638
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0061928, 0.0061625
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0197681, 0.0196812
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0052886, 0.0053142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090027, upper bound: 0.0090455
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090754, upper bound: 0.0089636
time: 0.57 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.36 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.36
Output dim: 8, lower bound: -0.0089636, upper bound: 0.0090754
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.36
Output dim: 8, lower bound: -0.0090455, upper bound: 0.0090027
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.36
Output dim: 8, lower bound: -0.0090048, upper bound: 0.0090394
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.36
Output dim: 8, lower bound: -0.0090757, upper bound: 0.0089587
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.36
Output dim: 8, lower bound: -0.0089532, upper bound: 0.0090861
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.36
Output dim: 8, lower bound: -0.0090308, upper bound: 0.0090084
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.36
Output dim: 8, lower bound: -0.0089960, upper bound: 0.0090559
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.36
Output dim: 8, lower bound: -0.0090593, upper bound: 0.0089681
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.36
Output dim: 8, lower bound: -0.0089681, upper bound: 0.0090593
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.36
Output dim: 8, lower bound: -0.0090559, upper bound: 0.0089960
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.36
Output dim: 8, lower bound: -0.0090084, upper bound: 0.0090308
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.36
Output dim: 8, lower bound: -0.0090861, upper bound: 0.0089532
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.36
Output dim: 8, lower bound: -0.0089587, upper bound: 0.0090757
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.36
Output dim: 8, lower bound: -0.0090394, upper bound: 0.0090048
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.36
Output dim: 8, lower bound: -0.0090027, upper bound: 0.0090455
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.36
Output dim: 8, lower bound: -0.0090754, upper bound: 0.0089636
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.36
Output dim: 8, lower bound: -0.0089636, upper bound: 0.0090754
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.36
Output dim: 8, lower bound: -0.0090455, upper bound: 0.0090027
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.36
Output dim: 8, lower bound: -0.0090048, upper bound: 0.0090394
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.36
Output dim: 8, lower bound: -0.0090757, upper bound: 0.0089587
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.36
Output dim: 8, lower bound: -0.0089532, upper bound: 0.0090861
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.36
Output dim: 8, lower bound: -0.0090308, upper bound: 0.0090084
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.36
Output dim: 8, lower bound: -0.0089960, upper bound: 0.0090559
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.36
Output dim: 8, lower bound: -0.0090593, upper bound: 0.0089681
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.36
Output dim: 8, lower bound: -0.0089681, upper bound: 0.0090593
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.36
Output dim: 8, lower bound: -0.0090559, upper bound: 0.0089960
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.36
Output dim: 8, lower bound: -0.0090084, upper bound: 0.0090308
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.36
Output dim: 8, lower bound: -0.0090861, upper bound: 0.0089532
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.36
Output dim: 8, lower bound: -0.0089587, upper bound: 0.0090757
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.36
Output dim: 8, lower bound: -0.0090394, upper bound: 0.0090048
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.36
Output dim: 8, lower bound: -0.0090027, upper bound: 0.0090455
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.36
Output dim: 8, lower bound: -0.0090754, upper bound: 0.0089636

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011364, 0.0011637
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0027308, 0.0028587
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0040562, 0.0038647
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0029645, 0.0028205
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0036042, 0.0034713
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0029513, 0.0028075
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0055733, 0.0058854
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0182554, 0.0191497
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0050727, 0.0048099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089082, upper bound: 0.0089870
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088810, upper bound: 0.0090234
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011610, 0.0011389
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0028461, 0.0027425
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0038823, 0.0040374
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0028337, 0.0029503
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0034835, 0.0035911
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0028207, 0.0029371
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0058547, 0.0056019
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0190615, 0.0183374
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0048340, 0.0050468

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089890, upper bound: 0.0088818
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089830, upper bound: 0.0089525
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011391, 0.0011613
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0027435, 0.0028475
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0040395, 0.0038837
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0029519, 0.0028348
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0035926, 0.0034845
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0029387, 0.0028218
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0056042, 0.0058582
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0183440, 0.0190716
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0050498, 0.0048359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089542, upper bound: 0.0089667
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088886, upper bound: 0.0089860
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011638, 0.0011358
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0028592, 0.0027280
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0038604, 0.0040569
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0028173, 0.0029651
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0034684, 0.0036047
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0028043, 0.0029518
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0058866, 0.0055664
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0191530, 0.0182356
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0048041, 0.0050737

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090225, upper bound: 0.0088740
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089915, upper bound: 0.0089044
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011360, 0.0011644
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0027291, 0.0028617
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0040607, 0.0038621
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0029679, 0.0028185
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0036073, 0.0034695
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0029546, 0.0028056
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0055691, 0.0058927
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0182433, 0.0191705
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0050789, 0.0048063

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088989, upper bound: 0.0090037
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088640, upper bound: 0.0090335
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011603, 0.0011391
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0028429, 0.0027434
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0038835, 0.0040325
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0028347, 0.0029467
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0034844, 0.0035878
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0028217, 0.0029335
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0058468, 0.0056040
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0190390, 0.0183434
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0048358, 0.0050402

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089764, upper bound: 0.0088978
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089500, upper bound: 0.0089583
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011389, 0.0011619
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0027427, 0.0028504
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0040438, 0.0038825
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0029551, 0.0028339
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0035956, 0.0034837
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0029419, 0.0028209
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0056023, 0.0058651
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0183385, 0.0190915
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0050556, 0.0048343

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089458, upper bound: 0.0089939
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088740, upper bound: 0.0090003
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011632, 0.0011362
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0028564, 0.0027300
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0038635, 0.0040529
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0028196, 0.0029620
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0034705, 0.0036019
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0028066, 0.0029487
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0058799, 0.0055713
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0191339, 0.0182498
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0048082, 0.0050681

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090080, upper bound: 0.0088890
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089664, upper bound: 0.0089128
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011367, 0.0011631
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0027321, 0.0028557
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0040518, 0.0038666
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0029612, 0.0028219
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0036011, 0.0034727
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0029479, 0.0028090
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0055764, 0.0058781
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0182644, 0.0191288
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0050666, 0.0048125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089128, upper bound: 0.0089664
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088890, upper bound: 0.0090080
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011615, 0.0011383
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0028482, 0.0027398
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0038782, 0.0040405
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0028307, 0.0029527
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0034807, 0.0035933
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0028177, 0.0029395
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0058599, 0.0055954
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0190764, 0.0183186
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0048285, 0.0050512

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090003, upper bound: 0.0088740
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089939, upper bound: 0.0089458
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011397, 0.0011608
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0027461, 0.0028450
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0040357, 0.0038876
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0029491, 0.0028377
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0035900, 0.0034872
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0029359, 0.0028247
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0056107, 0.0058520
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0183625, 0.0190539
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0050446, 0.0048414

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089583, upper bound: 0.0089500
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088978, upper bound: 0.0089764
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011646, 0.0011354
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0028627, 0.0027261
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0038576, 0.0040622
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0028152, 0.0029690
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0034664, 0.0036084
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0028022, 0.0029558
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0058952, 0.0055618
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0191776, 0.0182225
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0048002, 0.0050809

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090335, upper bound: 0.0088640
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090037, upper bound: 0.0088989
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011362, 0.0011636
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0027299, 0.0028582
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0040556, 0.0038633
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0029640, 0.0028194
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0036037, 0.0034704
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0029508, 0.0028065
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0055710, 0.0058843
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0182489, 0.0191465
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0050718, 0.0048080

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089044, upper bound: 0.0089915
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088740, upper bound: 0.0090225
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011609, 0.0011385
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0028457, 0.0027408
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0038796, 0.0040367
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0028317, 0.0029499
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0034817, 0.0035907
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0028187, 0.0029367
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0058537, 0.0055976
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0190587, 0.0183251
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0048304, 0.0050460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089860, upper bound: 0.0088886
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089667, upper bound: 0.0089542
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011395, 0.0011614
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0027454, 0.0028478
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0040399, 0.0038866
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0029523, 0.0028369
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0035929, 0.0034865
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0029390, 0.0028239
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0056090, 0.0058588
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0183576, 0.0190735
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0050503, 0.0048399

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089525, upper bound: 0.0089830
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088818, upper bound: 0.0089890
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011640, 0.0011359
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0028600, 0.0027284
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0038611, 0.0040581
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0028178, 0.0029659
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0034689, 0.0036055
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0028048, 0.0029527
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0058885, 0.0055675
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0191585, 0.0182389
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0048050, 0.0050753

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090234, upper bound: 0.0088810
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089870, upper bound: 0.0089082
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011359, 0.0011641
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0027284, 0.0028604
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0040587, 0.0038611
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0029664, 0.0028178
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0036059, 0.0034689
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0029532, 0.0028048
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0055675, 0.0058895
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0182389, 0.0191614
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0050762, 0.0048050

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089082, upper bound: 0.0089870
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088810, upper bound: 0.0090234
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011614, 0.0011393
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0028478, 0.0027442
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0038848, 0.0040399
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0028356, 0.0029523
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0034853, 0.0035929
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0028226, 0.0029390
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0058588, 0.0056060
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0190735, 0.0183491
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0048374, 0.0050503

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089890, upper bound: 0.0088818
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089830, upper bound: 0.0089525
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011385, 0.0011617
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0027408, 0.0028492
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0040420, 0.0038796
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0029538, 0.0028317
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0035943, 0.0034817
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0029406, 0.0028187
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0055976, 0.0058623
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0183251, 0.0190833
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0050532, 0.0048304

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089542, upper bound: 0.0089667
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088886, upper bound: 0.0089860
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011636, 0.0011362
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0028582, 0.0027296
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0038629, 0.0040556
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0028192, 0.0029640
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0034701, 0.0036037
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0028062, 0.0029508
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0058843, 0.0055705
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0191465, 0.0182473
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0048075, 0.0050718

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090225, upper bound: 0.0088740
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089915, upper bound: 0.0089044
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011354, 0.0011647
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0027261, 0.0028633
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0040632, 0.0038576
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0029698, 0.0028152
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0036090, 0.0034664
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0029565, 0.0028022
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0055618, 0.0058968
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0182225, 0.0191822
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0050823, 0.0048002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088989, upper bound: 0.0090037
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088640, upper bound: 0.0090335
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011608, 0.0011395
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0028450, 0.0027450
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0038860, 0.0040357
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0028365, 0.0029491
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0034862, 0.0035900
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0028235, 0.0029359
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0058520, 0.0056081
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0190539, 0.0183551
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0048392, 0.0050446

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089764, upper bound: 0.0088978
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089500, upper bound: 0.0089583
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011383, 0.0011623
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0027398, 0.0028520
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0040463, 0.0038782
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0029570, 0.0028307
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0035973, 0.0034807
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0029438, 0.0028177
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0055954, 0.0058692
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0183186, 0.0191031
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0050590, 0.0048285

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089458, upper bound: 0.0089939
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088740, upper bound: 0.0090003
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011631, 0.0011366
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0028557, 0.0027316
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0038660, 0.0040518
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0028215, 0.0029612
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0034722, 0.0036011
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0028085, 0.0029479
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0058781, 0.0055754
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0191288, 0.0182614
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0048117, 0.0050666

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090080, upper bound: 0.0088890
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089664, upper bound: 0.0089128
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011362, 0.0011635
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0027300, 0.0028574
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0040543, 0.0038635
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0029630, 0.0028196
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0036028, 0.0034705
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0029498, 0.0028066
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0055713, 0.0058822
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0182498, 0.0191405
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0050700, 0.0048082

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089128, upper bound: 0.0089664
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088890, upper bound: 0.0090080
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011619, 0.0011387
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0028504, 0.0027415
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0038807, 0.0040438
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0028325, 0.0029551
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0034825, 0.0035956
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0028195, 0.0029419
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0058651, 0.0055994
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0190914, 0.0183303
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0048319, 0.0050556

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090003, upper bound: 0.0088740
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089939, upper bound: 0.0089458
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011391, 0.0011612
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0027434, 0.0028467
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0040382, 0.0038835
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0029510, 0.0028347
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0035917, 0.0034844
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0029378, 0.0028217
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0056040, 0.0058561
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0183434, 0.0190655
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0050480, 0.0048358

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089583, upper bound: 0.0089500
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088978, upper bound: 0.0089764
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011644, 0.0011358
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0028617, 0.0027277
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0038601, 0.0040607
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0028171, 0.0029679
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0034682, 0.0036073
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0028041, 0.0029546
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0058927, 0.0055659
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0191705, 0.0182341
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0048036, 0.0050789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090335, upper bound: 0.0088640
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090037, upper bound: 0.0088989
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011358, 0.0011640
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0027280, 0.0028599
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0040581, 0.0038604
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0029659, 0.0028173
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0036055, 0.0034684
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0029527, 0.0028043
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0055664, 0.0058884
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0182356, 0.0191582
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0050752, 0.0048041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089044, upper bound: 0.0089915
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088740, upper bound: 0.0090225
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011613, 0.0011389
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0028475, 0.0027424
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0038821, 0.0040395
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0028336, 0.0029519
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0034834, 0.0035926
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0028206, 0.0029387
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0058582, 0.0056017
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0190716, 0.0183367
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0048338, 0.0050498

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089860, upper bound: 0.0088886
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089667, upper bound: 0.0089542
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011389, 0.0011618
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0027425, 0.0028495
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0040424, 0.0038822
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0029541, 0.0028337
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0035946, 0.0034835
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0029409, 0.0028207
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0056019, 0.0058629
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0183374, 0.0190852
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0050538, 0.0048340

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089525, upper bound: 0.0089830
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088818, upper bound: 0.0089890
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011637, 0.0011363
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0028587, 0.0027301
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0038636, 0.0040562
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0028197, 0.0029645
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0034706, 0.0036042
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0028067, 0.0029513
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0058854, 0.0055716
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0191497, 0.0182505
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0048085, 0.0050727

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090234, upper bound: 0.0088810
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089870, upper bound: 0.0089082
time: 0.73 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 8.54 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0089082, upper bound: 0.0089870
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0088810, upper bound: 0.0090234
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0089890, upper bound: 0.0088818
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0089830, upper bound: 0.0089525
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0089542, upper bound: 0.0089667
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0088886, upper bound: 0.0089860
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0090225, upper bound: 0.0088740
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0089915, upper bound: 0.0089044
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0088989, upper bound: 0.0090037
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0088640, upper bound: 0.0090335
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0089764, upper bound: 0.0088978
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0089500, upper bound: 0.0089583
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0089458, upper bound: 0.0089939
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0088740, upper bound: 0.0090003
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0090080, upper bound: 0.0088890
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0089664, upper bound: 0.0089128
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0089128, upper bound: 0.0089664
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0088890, upper bound: 0.0090080
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0090003, upper bound: 0.0088740
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0089939, upper bound: 0.0089458
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0089583, upper bound: 0.0089500
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0088978, upper bound: 0.0089764
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0090335, upper bound: 0.0088640
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0090037, upper bound: 0.0088989
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0089044, upper bound: 0.0089915
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0088740, upper bound: 0.0090225
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0089860, upper bound: 0.0088886
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0089667, upper bound: 0.0089542
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0089525, upper bound: 0.0089830
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0088818, upper bound: 0.0089890
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0090234, upper bound: 0.0088810
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0089870, upper bound: 0.0089082
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0089082, upper bound: 0.0089870
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0088810, upper bound: 0.0090234
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0089890, upper bound: 0.0088818
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0089830, upper bound: 0.0089525
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0089542, upper bound: 0.0089667
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0088886, upper bound: 0.0089860
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0090225, upper bound: 0.0088740
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0089915, upper bound: 0.0089044
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0088989, upper bound: 0.0090037
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0088640, upper bound: 0.0090335
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0089764, upper bound: 0.0088978
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0089500, upper bound: 0.0089583
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0089458, upper bound: 0.0089939
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0088740, upper bound: 0.0090003
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0090080, upper bound: 0.0088890
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0089664, upper bound: 0.0089128
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0089128, upper bound: 0.0089664
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0088890, upper bound: 0.0090080
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0090003, upper bound: 0.0088740
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0089939, upper bound: 0.0089458
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0089583, upper bound: 0.0089500
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0088978, upper bound: 0.0089764
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0090335, upper bound: 0.0088640
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0090037, upper bound: 0.0088989
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0089044, upper bound: 0.0089915
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0088740, upper bound: 0.0090225
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0089860, upper bound: 0.0088886
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0089667, upper bound: 0.0089542
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0089525, upper bound: 0.0089830
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0088818, upper bound: 0.0089890
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0090234, upper bound: 0.0088810
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.54
Output dim: 8, lower bound: -0.0089870, upper bound: 0.0089082

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011288, 0.0011549
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0026943, 0.0028166
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0039933, 0.0038101
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0029172, 0.0027794
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0035605, 0.0034335
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0029040, 0.0027665
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0054844, 0.0057828
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0180006, 0.0188557
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0049863, 0.0047350

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 128
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 7.98 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0079861, upper bound: 0.0089009
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0088200, upper bound: 0.0079174
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011276, 0.0011561
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0026887, 0.0028221
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0040015, 0.0038017
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0029234, 0.0027731
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0035662, 0.0034277
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0029102, 0.0027602
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0054707, 0.0057962
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0179614, 0.0188941
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0049976, 0.0047235

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 128
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 7.88 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0079479, upper bound: 0.0089377
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0087924, upper bound: 0.0079508
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011533, 0.0011301
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0028093, 0.0027005
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0038193, 0.0039823
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0027863, 0.0029089
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0034399, 0.0035529
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0027734, 0.0028958
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0057649, 0.0054993
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0188044, 0.0180434
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0047476, 0.0049712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 128
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 7.99 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079861, upper bound: 0.0087932
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089027, upper bound: 0.0079174
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011522, 0.0011312
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0028040, 0.0027057
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0038271, 0.0039744
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0027922, 0.0029030
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0034452, 0.0035474
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0027793, 0.0028898
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0057520, 0.0055120
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0187675, 0.0180797
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0047583, 0.0049604

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 128
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 7.64 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0079479, upper bound: 0.0088646
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088965, upper bound: 0.0079508
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011312, 0.0011525
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0027056, 0.0028055
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0039765, 0.0038270
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0029046, 0.0027921
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0035489, 0.0034452
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0028915, 0.0027792
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0055119, 0.0057556
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0180794, 0.0187776
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0049634, 0.0047582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 128
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 7.72 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0079856, upper bound: 0.0088801
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088663, upper bound: 0.0079172
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011303, 0.0011534
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0027014, 0.0028095
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0039826, 0.0038207
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0029092, 0.0027874
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0035531, 0.0034408
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0028960, 0.0027745
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0055016, 0.0057655
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0180500, 0.0188060
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0049717, 0.0047495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 128
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 7.85 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0079457, upper bound: 0.0088998
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0088002, upper bound: 0.0079508
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011559, 0.0011270
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0028214, 0.0026859
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0037975, 0.0040004
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0027699, 0.0029225
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0034247, 0.0035654
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0027571, 0.0029093
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0057944, 0.0054638
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0188888, 0.0179416
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0047177, 0.0049961

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 128
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 7.72 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079856, upper bound: 0.0087849
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089368, upper bound: 0.0079172
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011550, 0.0011283
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0028171, 0.0026922
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0038070, 0.0039940
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0027771, 0.0029177
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0034313, 0.0035610
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0027642, 0.0029045
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0057840, 0.0054793
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0188590, 0.0179860
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0047307, 0.0049873

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 128
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 7.86 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079457, upper bound: 0.0088161
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089051, upper bound: 0.0079508
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011286, 0.0011555
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0026937, 0.0028196
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0039977, 0.0038092
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0029205, 0.0027787
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0035636, 0.0034328
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0029074, 0.0027658
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0054828, 0.0057901
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0179962, 0.0188765
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0049925, 0.0047337

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 128
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 7.75 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0079861, upper bound: 0.0089176
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0088106, upper bound: 0.0079180
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011272, 0.0011564
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0026870, 0.0028237
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0040038, 0.0037991
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0029251, 0.0027712
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0035679, 0.0034259
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0029120, 0.0027583
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0054665, 0.0058001
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0179493, 0.0189051
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0050008, 0.0047199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 128
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 7.64 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0079475, upper bound: 0.0089480
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0087747, upper bound: 0.0079515
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011526, 0.0011303
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0028057, 0.0027013
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0038206, 0.0039769
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0027873, 0.0029048
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0034407, 0.0035491
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0027744, 0.0028917
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0057561, 0.0055014
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0187791, 0.0180494
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0047494, 0.0049638

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 128
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 7.94 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079861, upper bound: 0.0088097
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088901, upper bound: 0.0079180
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011515, 0.0011313
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0028008, 0.0027062
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0038278, 0.0039696
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0027927, 0.0028993
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0034458, 0.0035441
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0027798, 0.0028862
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0057442, 0.0055132
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0187450, 0.0180832
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0047593, 0.0049538

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 128
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 7.89 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0079475, upper bound: 0.0088705
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088629, upper bound: 0.0079515
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011311, 0.0011531
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0027054, 0.0028083
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0039808, 0.0038267
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0029078, 0.0027919
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0035519, 0.0034450
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0028947, 0.0027790
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0055114, 0.0057625
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0180780, 0.0187975
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0049692, 0.0047577

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 128
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 7.88 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0079859, upper bound: 0.0089077
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0088579, upper bound: 0.0079180
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011301, 0.0011540
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0027006, 0.0028124
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0039870, 0.0038195
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0029124, 0.0027865
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0035562, 0.0034400
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0028993, 0.0027736
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0054997, 0.0057726
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0180445, 0.0188263
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0049777, 0.0047479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 128
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 7.76 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0079455, upper bound: 0.0089144
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0087848, upper bound: 0.0079514
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011556, 0.0011274
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0028200, 0.0026879
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0038005, 0.0039982
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0027722, 0.0029209
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0034268, 0.0035640
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0027593, 0.0029078
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0057909, 0.0054687
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0188790, 0.0179558
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0047218, 0.0049932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 128
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 7.74 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079859, upper bound: 0.0088006
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089219, upper bound: 0.0079180
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011544, 0.0011285
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0028144, 0.0026928
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0038078, 0.0039899
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0027777, 0.0029146
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0034319, 0.0035582
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0027648, 0.0029015
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0057773, 0.0054807
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0188399, 0.0179900
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0047319, 0.0049817

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 128
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 7.63 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079455, upper bound: 0.0088246
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088798, upper bound: 0.0079514
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011287, 0.0011543
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0026941, 0.0028137
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0039888, 0.0038097
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0029138, 0.0027791
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0035574, 0.0034332
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0029007, 0.0027662
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0054837, 0.0057755
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0179987, 0.0188348
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0049802, 0.0047344

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 128
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 7.75 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0079514, upper bound: 0.0088798
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0088246, upper bound: 0.0079455
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011279, 0.0011552
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0026900, 0.0028180
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0039953, 0.0038036
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0029187, 0.0027746
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0035620, 0.0034290
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0029056, 0.0027617
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0054738, 0.0057862
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0179704, 0.0188654
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0049892, 0.0047261

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 128
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 7.91 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0079180, upper bound: 0.0089219
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0088006, upper bound: 0.0079859
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011538, 0.0011295
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0028115, 0.0026978
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0038153, 0.0039856
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0027833, 0.0029114
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0034371, 0.0035552
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0027704, 0.0028983
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0057704, 0.0054928
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0188200, 0.0180246
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0047421, 0.0049758

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 128
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 7.70 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079514, upper bound: 0.0087848
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089144, upper bound: 0.0079455
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011527, 0.0011307
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0028062, 0.0027036
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0038239, 0.0039776
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0027898, 0.0029054
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0034431, 0.0035496
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0027769, 0.0028922
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0057572, 0.0055068
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0187824, 0.0180650
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0047539, 0.0049648

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 128
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 7.82 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079180, upper bound: 0.0088579
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089077, upper bound: 0.0079859
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005109, 0.0009900, -0.0005109, 0.0009900, -0.0011316, 0.0011520
1: -0.0010717, 0.0028388, -0.0010717, 0.0028388, -0.0027078, 0.0028029
2: 0.0120885, 0.0179449, 0.0120885, 0.0179449, -0.0039727, 0.0038302
3: -0.0015368, 0.0028669, -0.0015368, 0.0028669, -0.0029017, 0.0027946
4: -0.0057972, -0.0017352, -0.0057972, -0.0017352, -0.0035463, 0.0034474
5: 0.0064041, 0.0107999, 0.0064041, 0.0107999, -0.0028886, 0.0027816
6: 0.0075730, 0.0105162, 0.0075730, 0.0105162, -0.0029432, 0.0029432
7: -0.0218448, -0.0123022, -0.0218448, -0.0123022, -0.0055171, 0.0057494
8: 0.9612030, 0.9885438, 0.9612030, 0.9885438, -0.0180945, 0.0187599
9: 0.0010201, 0.0090556, 0.0010201, 0.0090556, -0.0049582, 0.0047626

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 2.96 + 598.38 = 601.35 seconds
