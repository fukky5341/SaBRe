## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00357444


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541)
1: (-0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915)
2: (0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252)
3: (0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0140018, 0.0140019)
4: (-0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502)
5: (0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583)
6: (0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953)
7: (-0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395)
8: (0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394)
9: (0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0075431, 0.0075431)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.82 + 1.51 = 3.33 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0037125, upper bound: 0.0037125

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037125, upper bound: 0.0037126
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037125, upper bound: 0.0037126
time: 0.59 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.36 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.36
Output dim: 2, lower bound: -0.0037125, upper bound: 0.0037126
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.36
Output dim: 2, lower bound: -0.0037125, upper bound: 0.0037126

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0135026, 0.0136108
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0075431, 0.0075431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037051, upper bound: 0.0037051
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037051, upper bound: 0.0037051
time: 0.58 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0136108, 0.0135026
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0075431, 0.0075431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037051, upper bound: 0.0037051
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037051, upper bound: 0.0037051
time: 0.60 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.06 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.06
Output dim: 2, lower bound: -0.0037051, upper bound: 0.0037051
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.06
Output dim: 2, lower bound: -0.0037051, upper bound: 0.0037051
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.06
Output dim: 2, lower bound: -0.0037051, upper bound: 0.0037051
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.06
Output dim: 2, lower bound: -0.0037051, upper bound: 0.0037051

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0128267, 0.0130674
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0075431, 0.0075431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 58

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0129552, 0.0129349
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0075431, 0.0075431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 58

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0129349, 0.0129552
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0075431, 0.0075431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 58

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0130674, 0.0128267
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0075431, 0.0075431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 58

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.59 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.05 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0127597, 0.0130108
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0075431, 0.0075431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0127701, 0.0130005
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0075431, 0.0075431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0128915, 0.0128783
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0075431, 0.0075431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0128986, 0.0128677
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0075431, 0.0075431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0128677, 0.0128986
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0075431, 0.0075431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0128783, 0.0128915
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0075431, 0.0075431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0130005, 0.0127701
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0075431, 0.0075431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0130108, 0.0127597
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0075431, 0.0075431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.63 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.22 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0107713, 0.0112601
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0074871, 0.0071589

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0109712, 0.0110224
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0073275, 0.0072931

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0107817, 0.0112479
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0074789, 0.0071659

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0109859, 0.0110121
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0073206, 0.0073030

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0109031, 0.0111113
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0073872, 0.0072474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0111344, 0.0108899
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0072385, 0.0074027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0109102, 0.0110945
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0073759, 0.0072522

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0111427, 0.0108793
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0072314, 0.0074082

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0108793, 0.0111427
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0074082, 0.0072314

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0110945, 0.0109102
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0072522, 0.0073759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0108899, 0.0111344
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0074027, 0.0072385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0111113, 0.0109031
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0072474, 0.0073872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0110121, 0.0109859
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0073030, 0.0073206

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0112479, 0.0107817
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0071659, 0.0074789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0110224, 0.0109712
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0072931, 0.0073275

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0112601, 0.0107713
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0071589, 0.0074871

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
time: 0.66 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.22 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 2, lower bound: -0.0036538, upper bound: 0.0036538

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0093779, 0.0099136
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0069482, 0.0065885

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0094259, 0.0098667
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0069167, 0.0066208

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0095778, 0.0096613
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0067788, 0.0067227

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0096375, 0.0096290
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0067571, 0.0067628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0093883, 0.0099005
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0069394, 0.0065955

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0094373, 0.0098545
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0069085, 0.0066284

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0095925, 0.0096516
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0067723, 0.0067326

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0096485, 0.0096187
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0067502, 0.0067702

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0095097, 0.0097859
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0068624, 0.0066771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0095347, 0.0097179
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0068168, 0.0066938

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0097410, 0.0095572
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0067089, 0.0068323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0097833, 0.0094965
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0066682, 0.0068607

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0095168, 0.0097718
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0068530, 0.0066818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0095445, 0.0097011
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0068055, 0.0067004

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0097493, 0.0095464
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0067017, 0.0068379

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0097952, 0.0094859
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0066611, 0.0068687

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0094859, 0.0097952
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0068687, 0.0066611

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0095464, 0.0097493
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0068379, 0.0067017

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0097011, 0.0095445
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0067004, 0.0068055

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0097718, 0.0095168
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0066818, 0.0068530

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0094965, 0.0097833
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0068607, 0.0066682

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0095572, 0.0097410
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0068323, 0.0067089

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0097179, 0.0095347
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0066938, 0.0068168

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0097859, 0.0095097
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0066771, 0.0068624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0096187, 0.0096485
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0067702, 0.0067502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0096516, 0.0095925
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0067326, 0.0067723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0098545, 0.0094373
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0066284, 0.0069085

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0099005, 0.0093883
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0065955, 0.0069394

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0096290, 0.0096375
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0067628, 0.0067571

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0096613, 0.0095778
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0067227, 0.0067788

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0098667, 0.0094259
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0066208, 0.0069167

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041058, -0.0031517, -0.0041058, -0.0031517, -0.0009541, 0.0009541
1: -0.0064169, -0.0037254, -0.0064169, -0.0037254, -0.0026915, 0.0026915
2: 0.9671695, 0.9712947, 0.9671695, 0.9712947, -0.0041252, 0.0041252
3: 0.0159067, 0.0345806, 0.0159067, 0.0345806, -0.0099136, 0.0093779
4: -0.0033231, -0.0014729, -0.0033231, -0.0014729, -0.0018502, 0.0018502
5: 0.0131889, 0.0153472, 0.0131889, 0.0153472, -0.0021583, 0.0021583
6: 0.0035448, 0.0051401, 0.0035448, 0.0051401, -0.0015953, 0.0015953
7: -0.0167401, -0.0119006, -0.0167401, -0.0119006, -0.0048395, 0.0048395
8: 0.0034483, 0.0072878, 0.0034483, 0.0072878, -0.0038394, 0.0038394
9: 0.0032893, 0.0108323, 0.0032893, 0.0108323, -0.0065886, 0.0069482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
time: 0.62 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.22 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 2, lower bound: -0.0035367, upper bound: 0.0035367

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.33 + 202.33 = 205.65 seconds
