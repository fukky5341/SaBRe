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
Threshold: 0.0148662


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560)
1: (-0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419)
2: (-0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430)
3: (-0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184)
4: (-0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047)
5: (-0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383)
6: (-0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939)
7: (-0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378)
8: (0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605)
9: (-0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.39 + 2.58 = 3.97 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0186564, upper bound: 0.0186564

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0186043, upper bound: 0.0186443
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0186443, upper bound: 0.0186043
time: 1.24 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.45 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.45
Output dim: 8, lower bound: -0.0186043, upper bound: 0.0186443
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.45
Output dim: 8, lower bound: -0.0186443, upper bound: 0.0186043

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0182951, upper bound: 0.0183285
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0182951, upper bound: 0.0183285
time: 1.36 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0185671, upper bound: 0.0185280
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0185281, upper bound: 0.0185280
time: 1.48 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.27 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.27
Output dim: 8, lower bound: -0.0182951, upper bound: 0.0183285
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.27
Output dim: 8, lower bound: -0.0182951, upper bound: 0.0183285
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.27
Output dim: 8, lower bound: -0.0185671, upper bound: 0.0185280
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.27
Output dim: 8, lower bound: -0.0185281, upper bound: 0.0185280

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0176067, upper bound: 0.0176377
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0176067, upper bound: 0.0176377
time: 1.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0178649, upper bound: 0.0179012
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0178649, upper bound: 0.0179012
time: 1.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0183897, upper bound: 0.0184449
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0184872, upper bound: 0.0183564
time: 1.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0184840, upper bound: 0.0183659
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0183659, upper bound: 0.0184475
time: 1.37 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.12 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.12
Output dim: 8, lower bound: -0.0176067, upper bound: 0.0176377
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.12
Output dim: 8, lower bound: -0.0176067, upper bound: 0.0176377
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.12
Output dim: 8, lower bound: -0.0178649, upper bound: 0.0179012
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.12
Output dim: 8, lower bound: -0.0178649, upper bound: 0.0179012
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.12
Output dim: 8, lower bound: -0.0183897, upper bound: 0.0184449
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.12
Output dim: 8, lower bound: -0.0184872, upper bound: 0.0183564
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.12
Output dim: 8, lower bound: -0.0184840, upper bound: 0.0183659
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.12
Output dim: 8, lower bound: -0.0183659, upper bound: 0.0184475

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0173079, upper bound: 0.0173458
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0173134, upper bound: 0.0173338
time: 2.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0175628, upper bound: 0.0175911
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0175628, upper bound: 0.0176268
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0175662, upper bound: 0.0175945
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0175662, upper bound: 0.0175945
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0178548, upper bound: 0.0178882
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0178548, upper bound: 0.0178892
time: 1.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0180163, upper bound: 0.0181029
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0180163, upper bound: 0.0181029
time: 1.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0181402, upper bound: 0.0180165
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0181402, upper bound: 0.0180165
time: 1.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0184774, upper bound: 0.0183611
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0184793, upper bound: 0.0183616
time: 1.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0180453, upper bound: 0.0181312
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0180453, upper bound: 0.0181312
time: 1.60 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.34 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.34
Output dim: 8, lower bound: -0.0173079, upper bound: 0.0173458
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.34
Output dim: 8, lower bound: -0.0173134, upper bound: 0.0173338
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.34
Output dim: 8, lower bound: -0.0175628, upper bound: 0.0175911
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.34
Output dim: 8, lower bound: -0.0175628, upper bound: 0.0176268
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.34
Output dim: 8, lower bound: -0.0175662, upper bound: 0.0175945
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.34
Output dim: 8, lower bound: -0.0175662, upper bound: 0.0175945
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.34
Output dim: 8, lower bound: -0.0178548, upper bound: 0.0178882
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.34
Output dim: 8, lower bound: -0.0178548, upper bound: 0.0178892
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.34
Output dim: 8, lower bound: -0.0180163, upper bound: 0.0181029
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.34
Output dim: 8, lower bound: -0.0180163, upper bound: 0.0181029
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.34
Output dim: 8, lower bound: -0.0181402, upper bound: 0.0180165
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.34
Output dim: 8, lower bound: -0.0181402, upper bound: 0.0180165
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.34
Output dim: 8, lower bound: -0.0184774, upper bound: 0.0183611
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.34
Output dim: 8, lower bound: -0.0184793, upper bound: 0.0183616
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.34
Output dim: 8, lower bound: -0.0180453, upper bound: 0.0181312
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.34
Output dim: 8, lower bound: -0.0180453, upper bound: 0.0181312

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 163

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0173068, upper bound: 0.0173458
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0173068, upper bound: 0.0173458
time: 1.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 163

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0173134, upper bound: 0.0173335
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0173130, upper bound: 0.0173338
time: 1.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0172928, upper bound: 0.0174538
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0172928, upper bound: 0.0173155
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0168250, upper bound: 0.0168741
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0168250, upper bound: 0.0168741
time: 1.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0171892, upper bound: 0.0172203
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0171892, upper bound: 0.0172203
time: 1.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0175662, upper bound: 0.0175257
time: 1.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0175100, upper bound: 0.0175945
time: 1.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0176748, upper bound: 0.0177558
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0177230, upper bound: 0.0176985
time: 1.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0178118, upper bound: 0.0178347
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0178118, upper bound: 0.0178892
time: 1.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0166168, upper bound: 0.0166416
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0166168, upper bound: 0.0166416
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0179927, upper bound: 0.0180841
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0179927, upper bound: 0.0180841
time: 1.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0181214, upper bound: 0.0179936
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0181221, upper bound: 0.0179945
time: 1.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0178543, upper bound: 0.0178842
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0180189, upper bound: 0.0177419
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0177832, upper bound: 0.0176647
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0177832, upper bound: 0.0176647
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 163

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0184793, upper bound: 0.0183440
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0184048, upper bound: 0.0183616
time: 1.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0169138, upper bound: 0.0169509
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0169138, upper bound: 0.0169509
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 163

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0180796, upper bound: 0.0181292
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0180777, upper bound: 0.0181312
time: 1.49 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.17 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.17
Output dim: 8, lower bound: -0.0173068, upper bound: 0.0173458
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.17
Output dim: 8, lower bound: -0.0173068, upper bound: 0.0173458
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.17
Output dim: 8, lower bound: -0.0173134, upper bound: 0.0173335
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.17
Output dim: 8, lower bound: -0.0173130, upper bound: 0.0173338
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.17
Output dim: 8, lower bound: -0.0172928, upper bound: 0.0174538
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.17
Output dim: 8, lower bound: -0.0172928, upper bound: 0.0173155
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.17
Output dim: 8, lower bound: -0.0168250, upper bound: 0.0168741
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.17
Output dim: 8, lower bound: -0.0168250, upper bound: 0.0168741
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.17
Output dim: 8, lower bound: -0.0171892, upper bound: 0.0172203
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.17
Output dim: 8, lower bound: -0.0171892, upper bound: 0.0172203
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.17
Output dim: 8, lower bound: -0.0175662, upper bound: 0.0175257
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.17
Output dim: 8, lower bound: -0.0175100, upper bound: 0.0175945
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.17
Output dim: 8, lower bound: -0.0176748, upper bound: 0.0177558
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.17
Output dim: 8, lower bound: -0.0177230, upper bound: 0.0176985
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.17
Output dim: 8, lower bound: -0.0178118, upper bound: 0.0178347
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.17
Output dim: 8, lower bound: -0.0178118, upper bound: 0.0178892
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.17
Output dim: 8, lower bound: -0.0166168, upper bound: 0.0166416
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.17
Output dim: 8, lower bound: -0.0166168, upper bound: 0.0166416
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.17
Output dim: 8, lower bound: -0.0179927, upper bound: 0.0180841
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.17
Output dim: 8, lower bound: -0.0179927, upper bound: 0.0180841
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.17
Output dim: 8, lower bound: -0.0181214, upper bound: 0.0179936
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.17
Output dim: 8, lower bound: -0.0181221, upper bound: 0.0179945
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.17
Output dim: 8, lower bound: -0.0178543, upper bound: 0.0178842
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.17
Output dim: 8, lower bound: -0.0180189, upper bound: 0.0177419
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.17
Output dim: 8, lower bound: -0.0177832, upper bound: 0.0176647
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.17
Output dim: 8, lower bound: -0.0177832, upper bound: 0.0176647
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.17
Output dim: 8, lower bound: -0.0184793, upper bound: 0.0183440
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.17
Output dim: 8, lower bound: -0.0184048, upper bound: 0.0183616
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.17
Output dim: 8, lower bound: -0.0169138, upper bound: 0.0169509
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.17
Output dim: 8, lower bound: -0.0169138, upper bound: 0.0169509
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.17
Output dim: 8, lower bound: -0.0180796, upper bound: 0.0181292
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.17
Output dim: 8, lower bound: -0.0180777, upper bound: 0.0181312

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0170357, upper bound: 0.0170722
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0170357, upper bound: 0.0170722
time: 1.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 117

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0170464, upper bound: 0.0172067
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0171684, upper bound: 0.0170838
time: 1.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0168054, upper bound: 0.0169195
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0168888, upper bound: 0.0167999
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0164549, upper bound: 0.0164869
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0164549, upper bound: 0.0164869
time: 1.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0170359, upper bound: 0.0171459
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0170359, upper bound: 0.0171459
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0171620, upper bound: 0.0170293
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0171702, upper bound: 0.0170212
time: 1.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0166849, upper bound: 0.0167781
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0166849, upper bound: 0.0167292
time: 1.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0166581, upper bound: 0.0167591
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0167107, upper bound: 0.0167081
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0171058, upper bound: 0.0171363
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0171058, upper bound: 0.0171364
time: 2.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0166926, upper bound: 0.0168118
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0166926, upper bound: 0.0167080
time: 1.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0174286, upper bound: 0.0174427
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0174286, upper bound: 0.0174450
time: 1.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 163

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0174989, upper bound: 0.0175945
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0175100, upper bound: 0.0175873
time: 1.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 163

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0176742, upper bound: 0.0177558
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0176742, upper bound: 0.0177523
time: 1.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 163

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0176525, upper bound: 0.0176193
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0176486, upper bound: 0.0176313
time: 1.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0175823, upper bound: 0.0176988
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0175457, upper bound: 0.0175720
time: 1.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0176297, upper bound: 0.0177569
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0176828, upper bound: 0.0176995
time: 1.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0164025, upper bound: 0.0165061
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0164796, upper bound: 0.0164264
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0166013, upper bound: 0.0166207
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0166010, upper bound: 0.0166208
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0179799, upper bound: 0.0180677
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0179800, upper bound: 0.0180677
time: 1.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0166161, upper bound: 0.0166408
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0166161, upper bound: 0.0166408
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0181110, upper bound: 0.0179585
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0179567, upper bound: 0.0179834
time: 1.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 163

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0179915, upper bound: 0.0179945
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0179915, upper bound: 0.0179927
time: 1.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051023, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 117

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0173983, upper bound: 0.0175112
time: 1.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0173824, upper bound: 0.0175112
time: 1.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0050896
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0177307, upper bound: 0.0177307
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0177307, upper bound: 0.0177307
time: 1.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0176239, upper bound: 0.0175809
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0176888, upper bound: 0.0174963
time: 1.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0177730, upper bound: 0.0176389
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0177269, upper bound: 0.0176521
time: 1.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0050780
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 163

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0184793, upper bound: 0.0183433
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0183417, upper bound: 0.0183440
time: 1.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0050947, 0.0050937
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0179293, upper bound: 0.0179544
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0179293, upper bound: 0.0179544
time: 1.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0168606, upper bound: 0.0168946
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0168588, upper bound: 0.0168944
time: 1.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0158127, upper bound: 0.0158451
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0158127, upper bound: 0.0158451
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0177399, upper bound: 0.0177780
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0177399, upper bound: 0.0177780
time: 1.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0176203, upper bound: 0.0177098
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0176203, upper bound: 0.0177098
time: 1.45 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.27 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0170357, upper bound: 0.0170722
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0170357, upper bound: 0.0170722
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0170464, upper bound: 0.0172067
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0171684, upper bound: 0.0170838
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0168054, upper bound: 0.0169195
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0168888, upper bound: 0.0167999
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0164549, upper bound: 0.0164869
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0164549, upper bound: 0.0164869
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0170359, upper bound: 0.0171459
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0170359, upper bound: 0.0171459
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0171620, upper bound: 0.0170293
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0171702, upper bound: 0.0170212
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0166849, upper bound: 0.0167781
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0166849, upper bound: 0.0167292
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0166581, upper bound: 0.0167591
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0167107, upper bound: 0.0167081
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0171058, upper bound: 0.0171363
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0171058, upper bound: 0.0171364
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0166926, upper bound: 0.0168118
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0166926, upper bound: 0.0167080
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0174286, upper bound: 0.0174427
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0174286, upper bound: 0.0174450
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0174989, upper bound: 0.0175945
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0175100, upper bound: 0.0175873
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0176742, upper bound: 0.0177558
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0176742, upper bound: 0.0177523
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0176525, upper bound: 0.0176193
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0176486, upper bound: 0.0176313
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0175823, upper bound: 0.0176988
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0175457, upper bound: 0.0175720
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0176297, upper bound: 0.0177569
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0176828, upper bound: 0.0176995
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0164025, upper bound: 0.0165061
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0164796, upper bound: 0.0164264
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0166013, upper bound: 0.0166207
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0166010, upper bound: 0.0166208
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0179799, upper bound: 0.0180677
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0179800, upper bound: 0.0180677
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0166161, upper bound: 0.0166408
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0166161, upper bound: 0.0166408
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0181110, upper bound: 0.0179585
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0179567, upper bound: 0.0179834
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0179915, upper bound: 0.0179945
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0179915, upper bound: 0.0179927
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0173983, upper bound: 0.0175112
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0173824, upper bound: 0.0175112
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0177307, upper bound: 0.0177307
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0177307, upper bound: 0.0177307
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0176239, upper bound: 0.0175809
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0176888, upper bound: 0.0174963
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0177730, upper bound: 0.0176389
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0177269, upper bound: 0.0176521
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0184793, upper bound: 0.0183433
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0183417, upper bound: 0.0183440
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0179293, upper bound: 0.0179544
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0179293, upper bound: 0.0179544
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0168606, upper bound: 0.0168946
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0168588, upper bound: 0.0168944
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0158127, upper bound: 0.0158451
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0158127, upper bound: 0.0158451
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0177399, upper bound: 0.0177780
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0177399, upper bound: 0.0177780
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0176203, upper bound: 0.0177098
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 8, lower bound: -0.0176203, upper bound: 0.0177098

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0168768, upper bound: 0.0169878
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0168768, upper bound: 0.0169058
time: 1.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0169522, upper bound: 0.0169885
time: 1.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0169522, upper bound: 0.0169908
time: 1.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0050539, 0.0050994
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0169684, upper bound: 0.0171300
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0169684, upper bound: 0.0171300
time: 2.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0050825, 0.0050718
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0169856, upper bound: 0.0170220
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0169856, upper bound: 0.0170334
time: 1.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0168011, upper bound: 0.0169151
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0168010, upper bound: 0.0169151
time: 1.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0050937
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0167429, upper bound: 0.0167423
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0167429, upper bound: 0.0167462
time: 1.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0158042, upper bound: 0.0158518
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0158333, upper bound: 0.0158183
time: 1.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 117

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0163018, upper bound: 0.0163711
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0163417, upper bound: 0.0163284
time: 1.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 163

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0169674, upper bound: 0.0171059
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0169674, upper bound: 0.0171459
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0169674, upper bound: 0.0171059
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0169674, upper bound: 0.0171459
time: 1.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0050941, 0.0050912
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0163022, upper bound: 0.0162070
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0163022, upper bound: 0.0162070
time: 1.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0050682
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0170014, upper bound: 0.0170166
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0170010, upper bound: 0.0170170
time: 4.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0166796, upper bound: 0.0167678
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0166798, upper bound: 0.0167692
time: 1.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0166341, upper bound: 0.0165436
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0165566, upper bound: 0.0166447
time: 1.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0165850, upper bound: 0.0166744
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0165707, upper bound: 0.0166798
time: 1.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0160854, upper bound: 0.0161146
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0161180, upper bound: 0.0160809
time: 1.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0170469, upper bound: 0.0170773
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0170469, upper bound: 0.0170776
time: 1.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0170469, upper bound: 0.0170773
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0170469, upper bound: 0.0170776
time: 1.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0166104, upper bound: 0.0167322
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0166104, upper bound: 0.0167322
time: 1.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0166483, upper bound: 0.0166581
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0166483, upper bound: 0.0166633
time: 2.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0169672, upper bound: 0.0169798
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0169672, upper bound: 0.0169798
time: 1.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0169672, upper bound: 0.0169880
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0169672, upper bound: 0.0169880
time: 1.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0174967, upper bound: 0.0175876
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0174981, upper bound: 0.0175878
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0173581, upper bound: 0.0174990
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0173581, upper bound: 0.0174281
time: 1.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0175959, upper bound: 0.0176819
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0175959, upper bound: 0.0176864
time: 1.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0176519, upper bound: 0.0177285
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0176495, upper bound: 0.0177281
time: 1.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0157976, upper bound: 0.0157809
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0157976, upper bound: 0.0157809
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0174869, upper bound: 0.0175374
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0175585, upper bound: 0.0174943
time: 1.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0050325, 0.0050500
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0174743, upper bound: 0.0176322
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0174743, upper bound: 0.0176401
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0050631, 0.0050215
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0175657, upper bound: 0.0174824
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0176463, upper bound: 0.0174285
time: 1.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0050673, 0.0050890
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0175286, upper bound: 0.0175915
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0174741, upper bound: 0.0176651
time: 1.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0050708, 0.0050881
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 163

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0176274, upper bound: 0.0176995
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0176274, upper bound: 0.0176987
time: 1.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0050550, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0162913, upper bound: 0.0163409
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0162518, upper bound: 0.0164019
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0050841, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 117

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0159750, upper bound: 0.0159701
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0160273, upper bound: 0.0159367
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0161102, upper bound: 0.0161831
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0161629, upper bound: 0.0161379
time: 1.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0163878, upper bound: 0.0164875
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0164661, upper bound: 0.0164067
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0178433, upper bound: 0.0179668
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0178989, upper bound: 0.0178744
time: 1.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0176144, upper bound: 0.0176824
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0176144, upper bound: 0.0176824
time: 1.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0165321, upper bound: 0.0165474
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0165275, upper bound: 0.0165479
time: 1.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0165998, upper bound: 0.0166194
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0165997, upper bound: 0.0166197
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0179290, upper bound: 0.0178393
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0179915, upper bound: 0.0177996
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0179399, upper bound: 0.0179567
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0180181, upper bound: 0.0179834
time: 1.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0174875, upper bound: 0.0175727
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0174875, upper bound: 0.0174899
time: 1.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0174875, upper bound: 0.0175691
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0174875, upper bound: 0.0174890
time: 1.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0050752, 0.0050878
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0174714, upper bound: 0.0174734
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0174226, upper bound: 0.0175112
time: 1.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0050741, 0.0050904
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0168593, upper bound: 0.0170599
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0168593, upper bound: 0.0169798
time: 1.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0050783
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0177057, upper bound: 0.0174525
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0177057, upper bound: 0.0174525
time: 1.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0050804
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0177060, upper bound: 0.0177075
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0177060, upper bound: 0.0177307
time: 2.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0050820, 0.0050763
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0173937, upper bound: 0.0173498
time: 2.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0172840, upper bound: 0.0173498
time: 1.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0050964, 0.0050605
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0171834, upper bound: 0.0170906
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0172925, upper bound: 0.0170036
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0170613, upper bound: 0.0169442
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0170613, upper bound: 0.0169442
time: 1.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0172464, upper bound: 0.0172635
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0173176, upper bound: 0.0172636
time: 1.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0050665
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0179926, upper bound: 0.0179945
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0179926, upper bound: 0.0179945
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0050681
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0180713, upper bound: 0.0181974
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0180713, upper bound: 0.0180740
time: 1.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0050435, 0.0050311
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0177724, upper bound: 0.0178445
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0178892, upper bound: 0.0178114
time: 1.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0050324, 0.0050425
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0179363, upper bound: 0.0178967
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0178672, upper bound: 0.0179029
time: 1.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0168606, upper bound: 0.0168590
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0168457, upper bound: 0.0168946
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0167145, upper bound: 0.0168038
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0167666, upper bound: 0.0167455
time: 1.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152608, upper bound: 0.0152815
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152608, upper bound: 0.0152815
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0156233, upper bound: 0.0156969
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0156595, upper bound: 0.0156466
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0176764, upper bound: 0.0177105
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0176764, upper bound: 0.0177780
time: 1.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0176764, upper bound: 0.0177104
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0176764, upper bound: 0.0177780
time: 1.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0174481, upper bound: 0.0175760
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0174481, upper bound: 0.0175298
time: 1.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 117

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0173522, upper bound: 0.0175868
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0175425, upper bound: 0.0174384
time: 1.41 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 4.25 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0168768, upper bound: 0.0169878
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0168768, upper bound: 0.0169058
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0169522, upper bound: 0.0169885
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0169522, upper bound: 0.0169908
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0169684, upper bound: 0.0171300
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0169684, upper bound: 0.0171300
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0169856, upper bound: 0.0170220
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0169856, upper bound: 0.0170334
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0168011, upper bound: 0.0169151
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0168010, upper bound: 0.0169151
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0167429, upper bound: 0.0167423
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0167429, upper bound: 0.0167462
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0158042, upper bound: 0.0158518
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0158333, upper bound: 0.0158183
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0163018, upper bound: 0.0163711
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0163417, upper bound: 0.0163284
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0169674, upper bound: 0.0171059
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0169674, upper bound: 0.0171459
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0169674, upper bound: 0.0171059
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0169674, upper bound: 0.0171459
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0163022, upper bound: 0.0162070
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0163022, upper bound: 0.0162070
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0170014, upper bound: 0.0170166
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0170010, upper bound: 0.0170170
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0166796, upper bound: 0.0167678
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0166798, upper bound: 0.0167692
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0166341, upper bound: 0.0165436
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0165566, upper bound: 0.0166447
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0165850, upper bound: 0.0166744
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0165707, upper bound: 0.0166798
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0160854, upper bound: 0.0161146
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0161180, upper bound: 0.0160809
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0170469, upper bound: 0.0170773
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0170469, upper bound: 0.0170776
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0170469, upper bound: 0.0170773
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0170469, upper bound: 0.0170776
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0166104, upper bound: 0.0167322
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0166104, upper bound: 0.0167322
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0166483, upper bound: 0.0166581
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0166483, upper bound: 0.0166633
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0169672, upper bound: 0.0169798
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0169672, upper bound: 0.0169798
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0169672, upper bound: 0.0169880
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0169672, upper bound: 0.0169880
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0174967, upper bound: 0.0175876
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0174981, upper bound: 0.0175878
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0173581, upper bound: 0.0174990
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0173581, upper bound: 0.0174281
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0175959, upper bound: 0.0176819
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0175959, upper bound: 0.0176864
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0176519, upper bound: 0.0177285
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0176495, upper bound: 0.0177281
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0157976, upper bound: 0.0157809
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0157976, upper bound: 0.0157809
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0174869, upper bound: 0.0175374
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0175585, upper bound: 0.0174943
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0174743, upper bound: 0.0176322
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0174743, upper bound: 0.0176401
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0175657, upper bound: 0.0174824
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0176463, upper bound: 0.0174285
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0175286, upper bound: 0.0175915
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0174741, upper bound: 0.0176651
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0176274, upper bound: 0.0176995
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0176274, upper bound: 0.0176987
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0162913, upper bound: 0.0163409
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0162518, upper bound: 0.0164019
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0159750, upper bound: 0.0159701
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0160273, upper bound: 0.0159367
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0161102, upper bound: 0.0161831
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0161629, upper bound: 0.0161379
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0163878, upper bound: 0.0164875
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0164661, upper bound: 0.0164067
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0178433, upper bound: 0.0179668
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0178989, upper bound: 0.0178744
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0176144, upper bound: 0.0176824
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0176144, upper bound: 0.0176824
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0165321, upper bound: 0.0165474
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0165275, upper bound: 0.0165479
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0165998, upper bound: 0.0166194
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0165997, upper bound: 0.0166197
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0179290, upper bound: 0.0178393
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0179915, upper bound: 0.0177996
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0179399, upper bound: 0.0179567
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0180181, upper bound: 0.0179834
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0174875, upper bound: 0.0175727
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0174875, upper bound: 0.0174899
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0174875, upper bound: 0.0175691
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0174875, upper bound: 0.0174890
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0174714, upper bound: 0.0174734
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0174226, upper bound: 0.0175112
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0168593, upper bound: 0.0170599
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0168593, upper bound: 0.0169798
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0177057, upper bound: 0.0174525
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0177057, upper bound: 0.0174525
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0177060, upper bound: 0.0177075
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0177060, upper bound: 0.0177307
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0173937, upper bound: 0.0173498
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0172840, upper bound: 0.0173498
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0171834, upper bound: 0.0170906
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0172925, upper bound: 0.0170036
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0170613, upper bound: 0.0169442
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0170613, upper bound: 0.0169442
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0172464, upper bound: 0.0172635
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0173176, upper bound: 0.0172636
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0179926, upper bound: 0.0179945
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0179926, upper bound: 0.0179945
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0180713, upper bound: 0.0181974
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0180713, upper bound: 0.0180740
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0177724, upper bound: 0.0178445
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0178892, upper bound: 0.0178114
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0179363, upper bound: 0.0178967
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0178672, upper bound: 0.0179029
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0168606, upper bound: 0.0168590
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0168457, upper bound: 0.0168946
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0167145, upper bound: 0.0168038
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0167666, upper bound: 0.0167455
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0152608, upper bound: 0.0152815
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0152608, upper bound: 0.0152815
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0156233, upper bound: 0.0156969
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0156595, upper bound: 0.0156466
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0176764, upper bound: 0.0177105
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0176764, upper bound: 0.0177780
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0176764, upper bound: 0.0177104
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0176764, upper bound: 0.0177780
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0174481, upper bound: 0.0175760
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0174481, upper bound: 0.0175298
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0173522, upper bound: 0.0175868
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0175425, upper bound: 0.0174384

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051001, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0166082, upper bound: 0.0168475
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0167332, upper bound: 0.0167043
time: 1.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0166969, upper bound: 0.0167782
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0168185, upper bound: 0.0167329
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0168969, upper bound: 0.0169315
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0168952, upper bound: 0.0169330
time: 1.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0169490, upper bound: 0.0169865
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0169479, upper bound: 0.0169864
time: 1.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0050536, 0.0050991
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 117

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0168209, upper bound: 0.0170462
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0168209, upper bound: 0.0169714
time: 1.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0050536, 0.0050991
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0169490, upper bound: 0.0171144
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0169490, upper bound: 0.0171167
time: 1.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0050646, 0.0050464
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0168304, upper bound: 0.0169353
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0170247, upper bound: 0.0168574
time: 1.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0050572, 0.0050579
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0167072, upper bound: 0.0167425
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0168361, upper bound: 0.0167425
time: 1.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0167473, upper bound: 0.0168568
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0167386, upper bound: 0.0168619
time: 1.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010500, 0.0289060, 0.0010500, 0.0289060, -0.0278560, 0.0278560
1: -0.0066483, 0.0056935, -0.0066483, 0.0056935, -0.0123419, 0.0123419
2: -0.0059636, 0.0137794, -0.0059636, 0.0137794, -0.0197430, 0.0197430
3: -0.0043554, 0.0070630, -0.0043554, 0.0070630, -0.0114184, 0.0114184
4: -0.0054561, -0.0003513, -0.0054561, -0.0003513, -0.0051047, 0.0051047
5: -0.0043444, 0.0074939, -0.0043444, 0.0074939, -0.0118383, 0.0118383
6: -0.0128604, 0.0074335, -0.0128604, 0.0074335, -0.0202939, 0.0202939
7: -0.0210404, 0.0068974, -0.0210404, 0.0068974, -0.0279378, 0.0279378
8: 0.9770144, 1.0005749, 0.9770144, 1.0005749, -0.0235605, 0.0235605
9: -0.0163336, 0.0061821, -0.0163336, 0.0061821, -0.0225157, 0.0225157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0166096, upper bound: 0.0167459
time: 1.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0166096, upper bound: 0.0168313
time: 1.51 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 4.79 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.79
Output dim: 8, lower bound: -0.0166082, upper bound: 0.0168475
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.79
Output dim: 8, lower bound: -0.0167332, upper bound: 0.0167043
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.79
Output dim: 8, lower bound: -0.0166969, upper bound: 0.0167782
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.79
Output dim: 8, lower bound: -0.0168185, upper bound: 0.0167329
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.79
Output dim: 8, lower bound: -0.0168969, upper bound: 0.0169315
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.79
Output dim: 8, lower bound: -0.0168952, upper bound: 0.0169330
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.79
Output dim: 8, lower bound: -0.0169490, upper bound: 0.0169865
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.79
Output dim: 8, lower bound: -0.0169479, upper bound: 0.0169864
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.79
Output dim: 8, lower bound: -0.0168209, upper bound: 0.0170462
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.79
Output dim: 8, lower bound: -0.0168209, upper bound: 0.0169714
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.79
Output dim: 8, lower bound: -0.0169490, upper bound: 0.0171144
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.79
Output dim: 8, lower bound: -0.0169490, upper bound: 0.0171167
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.79
Output dim: 8, lower bound: -0.0168304, upper bound: 0.0169353
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.79
Output dim: 8, lower bound: -0.0170247, upper bound: 0.0168574
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.79
Output dim: 8, lower bound: -0.0167072, upper bound: 0.0167425
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.79
Output dim: 8, lower bound: -0.0168361, upper bound: 0.0167425
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.79
Output dim: 8, lower bound: -0.0167473, upper bound: 0.0168568
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.79
Output dim: 8, lower bound: -0.0167386, upper bound: 0.0168619
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.79
Output dim: 8, lower bound: -0.0166096, upper bound: 0.0167459
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.79
Output dim: 8, lower bound: -0.0166096, upper bound: 0.0168313
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0167429, upper bound: 0.0167423
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0167429, upper bound: 0.0167462
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0158042, upper bound: 0.0158518
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0158333, upper bound: 0.0158183
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0163018, upper bound: 0.0163711
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0163417, upper bound: 0.0163284
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0169674, upper bound: 0.0171059
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0169674, upper bound: 0.0171459
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0169674, upper bound: 0.0171059
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0169674, upper bound: 0.0171459
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0163022, upper bound: 0.0162070
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0163022, upper bound: 0.0162070
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0170014, upper bound: 0.0170166
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0170010, upper bound: 0.0170170
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0166796, upper bound: 0.0167678
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0166798, upper bound: 0.0167692
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0166341, upper bound: 0.0165436
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0165566, upper bound: 0.0166447
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0165850, upper bound: 0.0166744
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0165707, upper bound: 0.0166798
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0160854, upper bound: 0.0161146
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0161180, upper bound: 0.0160809
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0170469, upper bound: 0.0170773
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0170469, upper bound: 0.0170776
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0170469, upper bound: 0.0170773
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0170469, upper bound: 0.0170776
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0166104, upper bound: 0.0167322
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0166104, upper bound: 0.0167322
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0166483, upper bound: 0.0166581
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0166483, upper bound: 0.0166633
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0169672, upper bound: 0.0169798
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0169672, upper bound: 0.0169798
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0169672, upper bound: 0.0169880
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0169672, upper bound: 0.0169880
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0174967, upper bound: 0.0175876
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0174981, upper bound: 0.0175878
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0173581, upper bound: 0.0174990
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0173581, upper bound: 0.0174281
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0175959, upper bound: 0.0176819
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0175959, upper bound: 0.0176864
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0176519, upper bound: 0.0177285
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0176495, upper bound: 0.0177281
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0157976, upper bound: 0.0157809
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0157976, upper bound: 0.0157809
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0174869, upper bound: 0.0175374
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0175585, upper bound: 0.0174943
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0174743, upper bound: 0.0176322
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0174743, upper bound: 0.0176401
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0175657, upper bound: 0.0174824
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0176463, upper bound: 0.0174285
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0175286, upper bound: 0.0175915
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0174741, upper bound: 0.0176651
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0176274, upper bound: 0.0176995
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0176274, upper bound: 0.0176987
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0162913, upper bound: 0.0163409
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0162518, upper bound: 0.0164019
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0159750, upper bound: 0.0159701
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0160273, upper bound: 0.0159367
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0161102, upper bound: 0.0161831
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0161629, upper bound: 0.0161379
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0163878, upper bound: 0.0164875
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0164661, upper bound: 0.0164067
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0178433, upper bound: 0.0179668
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0178989, upper bound: 0.0178744
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0176144, upper bound: 0.0176824
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0176144, upper bound: 0.0176824
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0165321, upper bound: 0.0165474
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0165275, upper bound: 0.0165479
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0165998, upper bound: 0.0166194
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0165997, upper bound: 0.0166197
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0179290, upper bound: 0.0178393
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0179915, upper bound: 0.0177996
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0179399, upper bound: 0.0179567
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0180181, upper bound: 0.0179834
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0174875, upper bound: 0.0175727
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0174875, upper bound: 0.0174899
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0174875, upper bound: 0.0175691
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0174875, upper bound: 0.0174890
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0174714, upper bound: 0.0174734
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0174226, upper bound: 0.0175112
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0168593, upper bound: 0.0170599
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0168593, upper bound: 0.0169798
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0177057, upper bound: 0.0174525
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0177057, upper bound: 0.0174525
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0177060, upper bound: 0.0177075
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0177060, upper bound: 0.0177307
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0173937, upper bound: 0.0173498
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0172840, upper bound: 0.0173498
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0171834, upper bound: 0.0170906
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0172925, upper bound: 0.0170036
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0170613, upper bound: 0.0169442
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0170613, upper bound: 0.0169442
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0172464, upper bound: 0.0172635
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0173176, upper bound: 0.0172636
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0179926, upper bound: 0.0179945
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0179926, upper bound: 0.0179945
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0180713, upper bound: 0.0181974
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0180713, upper bound: 0.0180740
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0177724, upper bound: 0.0178445
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0178892, upper bound: 0.0178114
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0179363, upper bound: 0.0178967
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0178672, upper bound: 0.0179029
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0168606, upper bound: 0.0168590
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0168457, upper bound: 0.0168946
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0167145, upper bound: 0.0168038
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0167666, upper bound: 0.0167455
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0152608, upper bound: 0.0152815
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0152608, upper bound: 0.0152815
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0156233, upper bound: 0.0156969
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0156595, upper bound: 0.0156466
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0176764, upper bound: 0.0177105
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0176764, upper bound: 0.0177780
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0176764, upper bound: 0.0177104
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0176764, upper bound: 0.0177780
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0174481, upper bound: 0.0175760
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0174481, upper bound: 0.0175298
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0173522, upper bound: 0.0175868
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 8, lower bound: -0.0175425, upper bound: 0.0174384

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 3.97 + 596.46 = 600.43 seconds
