## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00031548


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0020188, 0.0020188)
1: (-0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0005692, 0.0005692)
2: (-0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0041995, 0.0041995)
3: (0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0005557, 0.0005557)
4: (0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0031385, 0.0031385)
5: (0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0008720, 0.0008720)
6: (0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0007915, 0.0007915)
7: (-0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0029536, 0.0029536)
8: (-0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0022988, 0.0022988)
9: (-0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001983, 0.0001983)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.33 + 1.77 = 3.11 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0005451, upper bound: 0.0005451

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005203, upper bound: 0.0005226
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005226, upper bound: 0.0005203
time: 0.91 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.06 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.06
Output dim: 5, lower bound: -0.0005203, upper bound: 0.0005226
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.06
Output dim: 5, lower bound: -0.0005226, upper bound: 0.0005203

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0018278, 0.0018237
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0005153, 0.0005142
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0038021, 0.0037937
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0005031, 0.0005020
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0028352, 0.0028414
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007877, 0.0007894
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0007150, 0.0007166
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0026682, 0.0026741
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0020813, 0.0020767
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001792, 0.0001796

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005110, upper bound: 0.0005060
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005051, upper bound: 0.0005135
time: 1.01 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0018237, 0.0018278
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0005142, 0.0005153
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0037937, 0.0038021
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0005020, 0.0005031
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0028414, 0.0028352
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007894, 0.0007877
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0007166, 0.0007150
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0026741, 0.0026682
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0020767, 0.0020813
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001796, 0.0001792

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004997, upper bound: 0.0004982
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005002, upper bound: 0.0004970
time: 0.87 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.92 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.92
Output dim: 5, lower bound: -0.0005110, upper bound: 0.0005060
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.92
Output dim: 5, lower bound: -0.0005051, upper bound: 0.0005135
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.92
Output dim: 5, lower bound: -0.0004997, upper bound: 0.0004982
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.92
Output dim: 5, lower bound: -0.0005002, upper bound: 0.0004970

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0017792, 0.0017838
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0005016, 0.0005029
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0037011, 0.0037106
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004898, 0.0004910
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0027731, 0.0027660
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007704, 0.0007685
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006993, 0.0006975
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0026098, 0.0026031
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0020260, 0.0020312
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001752, 0.0001748

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 205

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004956, upper bound: 0.0004964
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004956, upper bound: 0.0004997
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0017899, 0.0017752
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0005046, 0.0005005
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0037233, 0.0036928
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004927, 0.0004887
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0027597, 0.0027826
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007667, 0.0007731
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006960, 0.0007017
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0025972, 0.0026187
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0020381, 0.0020214
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001744, 0.0001758

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005049, upper bound: 0.0005101
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005014, upper bound: 0.0005133
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016126, 0.0016152
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004547, 0.0004554
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0033545, 0.0033599
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004439, 0.0004446
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025110, 0.0025070
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006976, 0.0006965
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006332, 0.0006322
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0023631, 0.0023593
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018363, 0.0018392
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001587, 0.0001584

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004872, upper bound: 0.0004847
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004862, upper bound: 0.0004859
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016112, 0.0016152
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004542, 0.0004554
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0033515, 0.0033600
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004435, 0.0004446
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025110, 0.0025047
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006976, 0.0006959
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006332, 0.0006317
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0023632, 0.0023572
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018346, 0.0018393
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001587, 0.0001583

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004747, upper bound: 0.0004788
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004819, upper bound: 0.0004726
time: 0.89 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.99 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 5, lower bound: -0.0004956, upper bound: 0.0004964
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 5, lower bound: -0.0004956, upper bound: 0.0004997
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 5, lower bound: -0.0005049, upper bound: 0.0005101
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 5, lower bound: -0.0005014, upper bound: 0.0005133
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 5, lower bound: -0.0004872, upper bound: 0.0004847
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 5, lower bound: -0.0004862, upper bound: 0.0004859
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 5, lower bound: -0.0004747, upper bound: 0.0004788
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 5, lower bound: -0.0004819, upper bound: 0.0004726

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0017410, 0.0017467
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004909, 0.0004925
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0036217, 0.0036335
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004793, 0.0004808
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0027154, 0.0027066
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007544, 0.0007520
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006848, 0.0006826
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0025555, 0.0025473
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019825, 0.0019890
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001716, 0.0001710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004794, upper bound: 0.0004803
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004794, upper bound: 0.0004883
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0017432, 0.0017456
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004915, 0.0004921
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0036262, 0.0036312
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004799, 0.0004805
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0027137, 0.0027100
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007539, 0.0007529
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006844, 0.0006834
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0025539, 0.0025504
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019850, 0.0019877
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001715, 0.0001713

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004822, upper bound: 0.0004857
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004822, upper bound: 0.0004875
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0017930, 0.0017795
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0005055, 0.0005017
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0037298, 0.0037017
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004936, 0.0004899
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0027664, 0.0027875
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007686, 0.0007744
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006977, 0.0007030
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0026035, 0.0026233
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0020417, 0.0020263
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001748, 0.0001761

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004724, upper bound: 0.0004718
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004694, upper bound: 0.0004760
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0017939, 0.0017783
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0005058, 0.0005014
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0037317, 0.0036993
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004938, 0.0004895
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0027646, 0.0027889
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007681, 0.0007748
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006972, 0.0007033
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0026018, 0.0026246
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0020428, 0.0020250
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001747, 0.0001762

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004933, upper bound: 0.0005012
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004902, upper bound: 0.0005056
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015550, 0.0015588
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004384, 0.0004395
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0032347, 0.0032426
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004281, 0.0004291
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0024233, 0.0024174
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006733, 0.0006716
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006111, 0.0006096
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0022806, 0.0022751
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017707, 0.0017750
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001531, 0.0001528

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003232, upper bound: 0.0003238
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003232, upper bound: 0.0003238
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015573, 0.0015576
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004391, 0.0004391
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0032396, 0.0032401
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004287, 0.0004288
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0024214, 0.0024211
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006727, 0.0006726
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006107, 0.0006106
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0022788, 0.0022785
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017733, 0.0017736
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001530, 0.0001530

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004819, upper bound: 0.0004735
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004732, upper bound: 0.0004815
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015883, 0.0015736
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004478, 0.0004437
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0033039, 0.0032734
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004372, 0.0004332
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0024464, 0.0024691
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006797, 0.0006860
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006169, 0.0006227
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0023023, 0.0023237
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018086, 0.0017919
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001546, 0.0001560

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004701, upper bound: 0.0004651
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004644, upper bound: 0.0004743
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015696, 0.0016152
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004425, 0.0004554
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0032650, 0.0033600
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004321, 0.0004446
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025110, 0.0024401
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006976, 0.0006779
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006332, 0.0006153
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0023632, 0.0022964
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017873, 0.0018393
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001587, 0.0001542

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004742, upper bound: 0.0004552
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004663, upper bound: 0.0004653
time: 0.91 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.04 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 5, lower bound: -0.0004794, upper bound: 0.0004803
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 5, lower bound: -0.0004794, upper bound: 0.0004883
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 5, lower bound: -0.0004822, upper bound: 0.0004857
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 5, lower bound: -0.0004822, upper bound: 0.0004875
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 5, lower bound: -0.0004724, upper bound: 0.0004718
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 5, lower bound: -0.0004694, upper bound: 0.0004760
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 5, lower bound: -0.0004933, upper bound: 0.0005012
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 5, lower bound: -0.0004902, upper bound: 0.0005056
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 5, lower bound: -0.0003232, upper bound: 0.0003238
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 5, lower bound: -0.0003232, upper bound: 0.0003238
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 5, lower bound: -0.0004819, upper bound: 0.0004735
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 5, lower bound: -0.0004732, upper bound: 0.0004815
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 5, lower bound: -0.0004701, upper bound: 0.0004651
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 5, lower bound: -0.0004644, upper bound: 0.0004743
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 5, lower bound: -0.0004742, upper bound: 0.0004552
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 5, lower bound: -0.0004663, upper bound: 0.0004653

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016827, 0.0016952
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004744, 0.0004779
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0035004, 0.0035263
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004632, 0.0004667
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026353, 0.0026160
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007322, 0.0007268
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006646, 0.0006597
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024801, 0.0024619
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019161, 0.0019303
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001665, 0.0001653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004889, upper bound: 0.0004675
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004842, upper bound: 0.0004725
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016884, 0.0016884
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004760, 0.0004760
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0035122, 0.0035121
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004648, 0.0004648
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026248, 0.0026248
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007292, 0.0007293
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006619, 0.0006619
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024702, 0.0024702
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019226, 0.0019225
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001659, 0.0001659

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004548, upper bound: 0.0004512
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004515, upper bound: 0.0004553
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016868, 0.0016926
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004756, 0.0004772
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0035088, 0.0035209
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004643, 0.0004659
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026313, 0.0026223
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007310, 0.0007285
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006636, 0.0006613
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024763, 0.0024678
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019207, 0.0019273
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001663, 0.0001657

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004557, upper bound: 0.0004498
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004515, upper bound: 0.0004523
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016881, 0.0016891
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004759, 0.0004762
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0035116, 0.0035137
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004647, 0.0004650
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026259, 0.0026244
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007296, 0.0007291
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006622, 0.0006618
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024713, 0.0024698
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019223, 0.0019234
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001659, 0.0001658

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004439, upper bound: 0.0004432
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004411, upper bound: 0.0004447
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0017623, 0.0017659
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004968, 0.0004979
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0036659, 0.0036735
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004851, 0.0004861
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0027453, 0.0027396
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007627, 0.0007612
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006923, 0.0006909
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0025837, 0.0025783
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0020067, 0.0020109
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001735, 0.0001731

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004686, upper bound: 0.0004588
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004578, upper bound: 0.0004677
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0017794, 0.0017795
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0005017, 0.0005017
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0037016, 0.0037017
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004898, 0.0004899
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0027664, 0.0027664
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007686, 0.0007686
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006977, 0.0006976
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0026035, 0.0026035
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0020263, 0.0020263
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001748, 0.0001748

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004095, upper bound: 0.0004130
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004095, upper bound: 0.0004130
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0017541, 0.0017465
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004945, 0.0004924
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0036488, 0.0036330
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004829, 0.0004808
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0027151, 0.0027269
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007543, 0.0007576
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006847, 0.0006877
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0025552, 0.0025663
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019974, 0.0019887
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001716, 0.0001723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004766, upper bound: 0.0004838
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004766, upper bound: 0.0004838
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0017626, 0.0017385
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004969, 0.0004901
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0036666, 0.0036164
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004852, 0.0004786
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0027027, 0.0027402
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007509, 0.0007613
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006816, 0.0006910
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0025435, 0.0025788
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0020071, 0.0019796
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001708, 0.0001732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004734, upper bound: 0.0004881
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004734, upper bound: 0.0004881
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015474, 0.0015564
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004363, 0.0004388
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0032189, 0.0032377
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004260, 0.0004285
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0024196, 0.0024056
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006722, 0.0006684
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006102, 0.0006067
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0022771, 0.0022640
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017620, 0.0017723
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001529, 0.0001520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003230, upper bound: 0.0003235
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003230, upper bound: 0.0003236
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015526, 0.0015588
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004377, 0.0004395
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0032298, 0.0032426
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004274, 0.0004291
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0024233, 0.0024137
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006733, 0.0006706
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006111, 0.0006087
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0022806, 0.0022716
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017680, 0.0017750
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001531, 0.0001525

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0003106, upper bound: 0.0003065
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0003056, upper bound: 0.0003117
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015433, 0.0015471
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004351, 0.0004362
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0032103, 0.0032182
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004248, 0.0004259
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0024051, 0.0023992
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006682, 0.0006666
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006065, 0.0006050
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0022635, 0.0022579
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017573, 0.0017617
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001520, 0.0001516

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004732, upper bound: 0.0004607
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004543, upper bound: 0.0004637
time: 1.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015473, 0.0015435
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004362, 0.0004352
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0032186, 0.0032108
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004259, 0.0004249
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0023996, 0.0024054
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006667, 0.0006683
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006051, 0.0006066
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0022583, 0.0022638
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017619, 0.0017576
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001516, 0.0001520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003802, upper bound: 0.0003869
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003802, upper bound: 0.0003869
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015766, 0.0015659
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004445, 0.0004415
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0032796, 0.0032575
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004340, 0.0004311
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0024344, 0.0024510
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006764, 0.0006809
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006139, 0.0006181
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0022911, 0.0023066
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017952, 0.0017831
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001538, 0.0001549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004664, upper bound: 0.0004391
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004487, upper bound: 0.0004620
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015810, 0.0015619
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004457, 0.0004404
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0032888, 0.0032491
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004352, 0.0004300
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0024282, 0.0024579
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006746, 0.0006829
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006124, 0.0006198
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0022852, 0.0023131
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018003, 0.0017786
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001534, 0.0001553

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004520, upper bound: 0.0004606
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004465, upper bound: 0.0004618
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0014961, 0.0015477
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004218, 0.0004363
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0031122, 0.0032194
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004119, 0.0004260
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0024060, 0.0023259
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006685, 0.0006462
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006068, 0.0005866
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0022643, 0.0021889
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017036, 0.0017623
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001520, 0.0001470

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004716, upper bound: 0.0004344
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004521, upper bound: 0.0004527
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015031, 0.0015427
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004238, 0.0004350
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0031267, 0.0032092
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004138, 0.0004247
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0023984, 0.0023367
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006663, 0.0006492
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006048, 0.0005893
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0022571, 0.0021991
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017116, 0.0017567
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001516, 0.0001477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003407, upper bound: 0.0003451
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003407, upper bound: 0.0003451
time: 0.89 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.96 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0004889, upper bound: 0.0004675
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0004842, upper bound: 0.0004725
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0004548, upper bound: 0.0004512
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0004515, upper bound: 0.0004553
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0004557, upper bound: 0.0004498
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0004515, upper bound: 0.0004523
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0004439, upper bound: 0.0004432
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0004411, upper bound: 0.0004447
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0004686, upper bound: 0.0004588
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0004578, upper bound: 0.0004677
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0004095, upper bound: 0.0004130
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0004095, upper bound: 0.0004130
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0004766, upper bound: 0.0004838
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0004766, upper bound: 0.0004838
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0004734, upper bound: 0.0004881
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0004734, upper bound: 0.0004881
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0003230, upper bound: 0.0003235
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0003230, upper bound: 0.0003236
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0003106, upper bound: 0.0003065
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0003056, upper bound: 0.0003117
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0004732, upper bound: 0.0004607
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0004543, upper bound: 0.0004637
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0003802, upper bound: 0.0003869
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0003802, upper bound: 0.0003869
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0004664, upper bound: 0.0004391
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0004487, upper bound: 0.0004620
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0004520, upper bound: 0.0004606
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0004465, upper bound: 0.0004618
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0004716, upper bound: 0.0004344
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0004521, upper bound: 0.0004527
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0003407, upper bound: 0.0003451
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0003407, upper bound: 0.0003451

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016458, 0.0016665
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004640, 0.0004699
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0034236, 0.0034667
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004531, 0.0004588
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025908, 0.0025586
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007198, 0.0007108
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006534, 0.0006452
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024383, 0.0024079
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018741, 0.0018977
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001637, 0.0001617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004566, upper bound: 0.0004349
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004522, upper bound: 0.0004379
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016532, 0.0016583
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004661, 0.0004675
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0034389, 0.0034495
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004551, 0.0004565
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025780, 0.0025700
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007162, 0.0007140
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006501, 0.0006481
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024262, 0.0024187
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018825, 0.0018883
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001629, 0.0001624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004652, upper bound: 0.0004537
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004650, upper bound: 0.0004537
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016567, 0.0016744
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004671, 0.0004721
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0034463, 0.0034830
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004561, 0.0004609
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026030, 0.0025756
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007232, 0.0007156
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006564, 0.0006495
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024497, 0.0024239
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018865, 0.0019066
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001645, 0.0001628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004479, upper bound: 0.0004417
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004427, upper bound: 0.0004431
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016744, 0.0016884
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004721, 0.0004760
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0034831, 0.0035121
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004609, 0.0004648
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026248, 0.0026030
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007292, 0.0007232
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006619, 0.0006564
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024702, 0.0024498
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019066, 0.0019225
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001659, 0.0001645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004477, upper bound: 0.0004408
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004389, upper bound: 0.0004512
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016541, 0.0016784
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004663, 0.0004732
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0034408, 0.0034913
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004553, 0.0004620
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026092, 0.0025714
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007249, 0.0007144
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006580, 0.0006485
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024555, 0.0024200
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018835, 0.0019112
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001649, 0.0001625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004536, upper bound: 0.0004371
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004389, upper bound: 0.0004476
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016726, 0.0016926
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004716, 0.0004772
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0034793, 0.0035209
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004604, 0.0004659
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026313, 0.0026002
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007310, 0.0007224
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006636, 0.0006557
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024763, 0.0024471
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019045, 0.0019273
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001663, 0.0001643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004440, upper bound: 0.0004422
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004408, upper bound: 0.0004445
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016836, 0.0016904
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004747, 0.0004766
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0035021, 0.0035163
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004635, 0.0004653
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026279, 0.0026173
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007301, 0.0007272
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006627, 0.0006600
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024731, 0.0024631
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019171, 0.0019248
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001661, 0.0001654

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004390, upper bound: 0.0004294
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004287, upper bound: 0.0004384
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016881, 0.0016846
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004759, 0.0004749
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0035116, 0.0035042
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004647, 0.0004637
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026188, 0.0026244
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007276, 0.0007291
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006604, 0.0006618
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024646, 0.0024698
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019223, 0.0019182
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001655, 0.0001658

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003255, upper bound: 0.0003276
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003255, upper bound: 0.0003276
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0017417, 0.0017586
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004910, 0.0004958
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0036230, 0.0036582
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004794, 0.0004841
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0027339, 0.0027076
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007596, 0.0007523
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006894, 0.0006828
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0025729, 0.0025482
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019832, 0.0020025
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001728, 0.0001711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004607, upper bound: 0.0004457
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004539, upper bound: 0.0004510
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0017540, 0.0017453
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004945, 0.0004921
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0036487, 0.0036306
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004828, 0.0004805
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0027133, 0.0027268
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007538, 0.0007576
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006843, 0.0006877
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0025535, 0.0025662
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019973, 0.0019874
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001715, 0.0001723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004345, upper bound: 0.0004473
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004384, upper bound: 0.0004470
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0017417, 0.0017270
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004910, 0.0004869
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0036230, 0.0035925
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004794, 0.0004754
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026848, 0.0027076
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007459, 0.0007523
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006771, 0.0006828
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0025267, 0.0025482
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019832, 0.0019665
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001697, 0.0001711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003318, upper bound: 0.0003312
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003318, upper bound: 0.0003312
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0017264, 0.0017795
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004867, 0.0005017
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0035913, 0.0037017
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004752, 0.0004899
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0027664, 0.0026839
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007686, 0.0007457
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006977, 0.0006768
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0026035, 0.0025258
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019659, 0.0020263
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001748, 0.0001696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004054, upper bound: 0.0003987
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003951, upper bound: 0.0004087
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0017327, 0.0017223
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004885, 0.0004856
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0036043, 0.0035827
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004770, 0.0004741
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026775, 0.0026937
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007439, 0.0007484
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006752, 0.0006793
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0025198, 0.0025350
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019730, 0.0019612
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001692, 0.0001702

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004696, upper bound: 0.0004765
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004696, upper bound: 0.0004773
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0017541, 0.0017251
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004945, 0.0004864
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0036488, 0.0035885
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004829, 0.0004749
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026818, 0.0027269
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007451, 0.0007576
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006763, 0.0006877
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0025239, 0.0025663
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019974, 0.0019644
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001695, 0.0001723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004720, upper bound: 0.0004696
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004604, upper bound: 0.0004790
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0017412, 0.0017162
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004909, 0.0004839
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0036221, 0.0035700
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004793, 0.0004724
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026680, 0.0027070
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007413, 0.0007521
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006728, 0.0006827
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0025109, 0.0025476
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019828, 0.0019542
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001686, 0.0001711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004516, upper bound: 0.0004666
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004515, upper bound: 0.0004666
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0017626, 0.0017171
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004969, 0.0004841
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0036666, 0.0035719
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004852, 0.0004727
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026694, 0.0027402
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007416, 0.0007613
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006732, 0.0006910
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0025122, 0.0025788
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0020071, 0.0019553
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001687, 0.0001732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004708, upper bound: 0.0004651
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004565, upper bound: 0.0004855
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015500, 0.0015598
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004370, 0.0004398
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0032242, 0.0032447
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004267, 0.0004294
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0024249, 0.0024096
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006737, 0.0006695
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006115, 0.0006077
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0022821, 0.0022677
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017649, 0.0017761
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001532, 0.0001523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 205

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003161, upper bound: 0.0003145
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003141, upper bound: 0.0003166
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015510, 0.0015590
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004373, 0.0004395
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0032264, 0.0032430
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004270, 0.0004292
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0024236, 0.0024112
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006733, 0.0006699
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006112, 0.0006081
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0022809, 0.0022692
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017661, 0.0017752
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001532, 0.0001524

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003183, upper bound: 0.0003087
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003074, upper bound: 0.0003187
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0014919, 0.0015063
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004206, 0.0004247
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0031034, 0.0031334
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004107, 0.0004147
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0023417, 0.0023193
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006506, 0.0006444
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0005906, 0.0005849
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0022038, 0.0021827
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0016988, 0.0017153
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001480, 0.0001466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004655, upper bound: 0.0004527
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004650, upper bound: 0.0004532
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0014976, 0.0014957
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004222, 0.0004217
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0031153, 0.0031113
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004123, 0.0004117
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0023252, 0.0023282
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006460, 0.0006468
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0005864, 0.0005871
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0021883, 0.0021911
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017053, 0.0017031
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001469, 0.0001471

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004636, upper bound: 0.0004584
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004492, upper bound: 0.0004635
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015395, 0.0015519
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004340, 0.0004376
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0032024, 0.0032284
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004238, 0.0004272
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0024127, 0.0023933
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006703, 0.0006649
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006084, 0.0006036
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0022706, 0.0022524
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017530, 0.0017672
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001525, 0.0001512

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002374, upper bound: 0.0002454
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002374, upper bound: 0.0002454
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015473, 0.0015357
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004362, 0.0004330
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0032186, 0.0031946
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004259, 0.0004228
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0023875, 0.0024054
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006633, 0.0006683
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006021, 0.0006066
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0022469, 0.0022638
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017619, 0.0017487
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001509, 0.0001520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003722, upper bound: 0.0003771
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003702, upper bound: 0.0003789
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015330, 0.0015333
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004322, 0.0004323
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0031889, 0.0031897
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004220, 0.0004221
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0023838, 0.0023832
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006623, 0.0006621
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006011, 0.0006010
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0022434, 0.0022428
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017456, 0.0017460
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001506, 0.0001506

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 205

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004601, upper bound: 0.0004284
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004576, upper bound: 0.0004328
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015460, 0.0015223
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004359, 0.0004292
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0032160, 0.0031668
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004256, 0.0004191
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0023666, 0.0024034
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006575, 0.0006677
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0005968, 0.0006061
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0022273, 0.0022619
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017604, 0.0017335
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001496, 0.0001519

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003403, upper bound: 0.0003380
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003403, upper bound: 0.0003380
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015223, 0.0015049
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004292, 0.0004243
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0031667, 0.0031304
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004191, 0.0004143
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0023395, 0.0023666
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006500, 0.0006575
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0005900, 0.0005968
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0022017, 0.0022273
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017335, 0.0017136
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001478, 0.0001496

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003225, upper bound: 0.0003250
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003225, upper bound: 0.0003250
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015252, 0.0015032
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004300, 0.0004238
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0031727, 0.0031270
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004199, 0.0004138
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0023369, 0.0023711
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006493, 0.0006587
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0005893, 0.0005979
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0021993, 0.0022314
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017367, 0.0017117
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001477, 0.0001498

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004290, upper bound: 0.0004396
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004290, upper bound: 0.0004397
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0014788, 0.0015433
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004169, 0.0004351
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0030762, 0.0032104
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004071, 0.0004248
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0023992, 0.0022989
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006666, 0.0006387
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006051, 0.0005798
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0022580, 0.0021636
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0016839, 0.0017574
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001516, 0.0001453

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004713, upper bound: 0.0004328
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004687, upper bound: 0.0004342
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0014892, 0.0015341
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004199, 0.0004325
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0030979, 0.0031913
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004100, 0.0004223
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0023850, 0.0023152
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006626, 0.0006432
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006015, 0.0005838
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0022445, 0.0021788
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0016958, 0.0017469
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001507, 0.0001463

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004441, upper bound: 0.0004397
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004424, upper bound: 0.0004458
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0014985, 0.0015434
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004225, 0.0004351
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0031173, 0.0032105
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004125, 0.0004249
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0023993, 0.0023297
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006666, 0.0006472
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006051, 0.0005875
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0022580, 0.0021925
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017064, 0.0017574
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001516, 0.0001472

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003405, upper bound: 0.0003420
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003380, upper bound: 0.0003449
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015031, 0.0015382
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004238, 0.0004337
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0031267, 0.0031998
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004138, 0.0004234
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0023913, 0.0023367
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006644, 0.0006492
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006031, 0.0005893
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0022505, 0.0021991
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017116, 0.0017516
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001511, 0.0001477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003327, upper bound: 0.0003369
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003327, upper bound: 0.0003369
time: 0.76 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.76 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004566, upper bound: 0.0004349
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004522, upper bound: 0.0004379
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004652, upper bound: 0.0004537
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004650, upper bound: 0.0004537
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004479, upper bound: 0.0004417
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004427, upper bound: 0.0004431
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004477, upper bound: 0.0004408
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004389, upper bound: 0.0004512
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004536, upper bound: 0.0004371
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004389, upper bound: 0.0004476
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004440, upper bound: 0.0004422
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004408, upper bound: 0.0004445
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004390, upper bound: 0.0004294
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004287, upper bound: 0.0004384
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0003255, upper bound: 0.0003276
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0003255, upper bound: 0.0003276
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004607, upper bound: 0.0004457
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004539, upper bound: 0.0004510
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004345, upper bound: 0.0004473
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004384, upper bound: 0.0004470
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0003318, upper bound: 0.0003312
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0003318, upper bound: 0.0003312
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004054, upper bound: 0.0003987
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0003951, upper bound: 0.0004087
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004696, upper bound: 0.0004765
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004696, upper bound: 0.0004773
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004720, upper bound: 0.0004696
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004604, upper bound: 0.0004790
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004516, upper bound: 0.0004666
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004515, upper bound: 0.0004666
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004708, upper bound: 0.0004651
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004565, upper bound: 0.0004855
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0003161, upper bound: 0.0003145
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0003141, upper bound: 0.0003166
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0003183, upper bound: 0.0003087
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0003074, upper bound: 0.0003187
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004655, upper bound: 0.0004527
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004650, upper bound: 0.0004532
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004636, upper bound: 0.0004584
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004492, upper bound: 0.0004635
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0002374, upper bound: 0.0002454
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0002374, upper bound: 0.0002454
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0003722, upper bound: 0.0003771
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0003702, upper bound: 0.0003789
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004601, upper bound: 0.0004284
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004576, upper bound: 0.0004328
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0003403, upper bound: 0.0003380
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0003403, upper bound: 0.0003380
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0003225, upper bound: 0.0003250
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0003225, upper bound: 0.0003250
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004290, upper bound: 0.0004396
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004290, upper bound: 0.0004397
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004713, upper bound: 0.0004328
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004687, upper bound: 0.0004342
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004441, upper bound: 0.0004397
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0004424, upper bound: 0.0004458
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0003405, upper bound: 0.0003420
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0003380, upper bound: 0.0003449
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0003327, upper bound: 0.0003369
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 5, lower bound: -0.0003327, upper bound: 0.0003369

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016133, 0.0016521
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004549, 0.0004658
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0033561, 0.0034366
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004441, 0.0004548
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025683, 0.0025081
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007136, 0.0006968
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006477, 0.0006325
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024171, 0.0023604
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018371, 0.0018812
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001623, 0.0001585

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004532, upper bound: 0.0004250
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004379, upper bound: 0.0004307
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016313, 0.0016665
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004599, 0.0004699
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0033935, 0.0034667
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004491, 0.0004588
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025908, 0.0025361
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007198, 0.0007046
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006534, 0.0006396
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024383, 0.0023867
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018576, 0.0018977
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001637, 0.0001603

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004500, upper bound: 0.0004245
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004388, upper bound: 0.0004357
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016333, 0.0016401
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004605, 0.0004624
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0033977, 0.0034116
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004496, 0.0004515
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025496, 0.0025392
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007084, 0.0007055
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006430, 0.0006403
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0023995, 0.0023897
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018599, 0.0018675
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001611, 0.0001605

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004608, upper bound: 0.0004428
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004505, upper bound: 0.0004490
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016532, 0.0016384
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004661, 0.0004619
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0034389, 0.0034083
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004551, 0.0004510
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025471, 0.0025700
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007077, 0.0007140
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006423, 0.0006481
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0023971, 0.0024187
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018825, 0.0018657
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001610, 0.0001624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004551, upper bound: 0.0004439
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004378, upper bound: 0.0004440
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016194, 0.0016466
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004566, 0.0004642
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0033686, 0.0034253
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004458, 0.0004533
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025598, 0.0025175
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007112, 0.0006994
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006455, 0.0006349
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024091, 0.0023692
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018440, 0.0018750
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001618, 0.0001591

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003912, upper bound: 0.0003858
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003912, upper bound: 0.0003858
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016255, 0.0016370
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004583, 0.0004615
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0033814, 0.0034053
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004475, 0.0004506
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025449, 0.0025271
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007070, 0.0007021
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006418, 0.0006373
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0023950, 0.0023783
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018510, 0.0018641
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001608, 0.0001597

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003963, upper bound: 0.0003967
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003963, upper bound: 0.0003967
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016562, 0.0016849
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004670, 0.0004750
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0034453, 0.0035050
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004559, 0.0004638
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026194, 0.0025748
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007278, 0.0007154
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006606, 0.0006493
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024652, 0.0024232
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018860, 0.0019187
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001655, 0.0001627

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004266, upper bound: 0.0004164
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004266, upper bound: 0.0004164
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016695, 0.0016707
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004707, 0.0004710
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0034729, 0.0034754
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004596, 0.0004599
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025973, 0.0025954
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007216, 0.0007211
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006550, 0.0006545
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024444, 0.0024426
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019011, 0.0019025
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001641, 0.0001640

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004164, upper bound: 0.0004315
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004192, upper bound: 0.0004301
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016272, 0.0016607
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004588, 0.0004682
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0033849, 0.0034545
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004479, 0.0004571
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025817, 0.0025296
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007173, 0.0007028
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006511, 0.0006379
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024297, 0.0023807
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018529, 0.0018910
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001631, 0.0001599

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004471, upper bound: 0.0004302
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004461, upper bound: 0.0004302
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016364, 0.0016498
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004614, 0.0004651
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0034040, 0.0034319
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004505, 0.0004542
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025648, 0.0025439
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007126, 0.0007068
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006468, 0.0006415
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024137, 0.0023941
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018634, 0.0018786
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001621, 0.0001608

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004338, upper bound: 0.0004391
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004290, upper bound: 0.0004425
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016337, 0.0016611
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004606, 0.0004683
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0033984, 0.0034554
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004497, 0.0004573
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025824, 0.0025397
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007175, 0.0007056
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006512, 0.0006405
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024303, 0.0023902
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018603, 0.0018915
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001632, 0.0001605

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004402, upper bound: 0.0004289
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004277, upper bound: 0.0004383
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016419, 0.0016537
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004629, 0.0004662
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0034154, 0.0034401
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004520, 0.0004552
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025709, 0.0025524
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007143, 0.0007091
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006483, 0.0006437
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024195, 0.0024021
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018696, 0.0018831
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001625, 0.0001613

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004345, upper bound: 0.0004377
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004282, upper bound: 0.0004379
time: 1.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016696, 0.0016808
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004707, 0.0004739
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0034731, 0.0034963
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004596, 0.0004627
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026129, 0.0025956
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007260, 0.0007211
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006589, 0.0006546
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024591, 0.0024427
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019012, 0.0019139
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001651, 0.0001640

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004352, upper bound: 0.0004146
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004207, upper bound: 0.0004254
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016732, 0.0016764
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004717, 0.0004726
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0034807, 0.0034873
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004606, 0.0004615
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026062, 0.0026012
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007241, 0.0007227
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006572, 0.0006560
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024527, 0.0024481
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019053, 0.0019090
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001647, 0.0001644

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001882, upper bound: 0.0001892
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001882, upper bound: 0.0001892
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016548, 0.0016338
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004666, 0.0004606
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0034424, 0.0033985
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004555, 0.0004497
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025399, 0.0025726
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007056, 0.0007147
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006405, 0.0006488
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0023903, 0.0024211
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018844, 0.0018604
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001605, 0.0001626

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0003109, upper bound: 0.0003129
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0003109, upper bound: 0.0003117
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016373, 0.0016846
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004616, 0.0004749
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0034059, 0.0035042
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004507, 0.0004637
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026188, 0.0025454
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007276, 0.0007072
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006604, 0.0006419
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024646, 0.0023955
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018644, 0.0019182
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001655, 0.0001609

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003253, upper bound: 0.0003246
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003236, upper bound: 0.0003275
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016856, 0.0017095
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004752, 0.0004820
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0035065, 0.0035560
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004640, 0.0004706
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026576, 0.0026205
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007384, 0.0007281
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006702, 0.0006609
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0025011, 0.0024662
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019194, 0.0019466
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001679, 0.0001656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004363, upper bound: 0.0004248
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004363, upper bound: 0.0004248
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016916, 0.0017025
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004769, 0.0004800
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0035188, 0.0035416
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004657, 0.0004687
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026468, 0.0026297
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007354, 0.0007306
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006675, 0.0006632
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024909, 0.0024749
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019262, 0.0019387
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001673, 0.0001662

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004377, upper bound: 0.0004443
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004471, upper bound: 0.0004443
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015126, 0.0015102
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004265, 0.0004258
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0031465, 0.0031414
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004164, 0.0004157
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0023477, 0.0023515
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006523, 0.0006533
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0005921, 0.0005930
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0022095, 0.0022130
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017224, 0.0017196
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001484, 0.0001486

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003795, upper bound: 0.0003936
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003795, upper bound: 0.0003936
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015189, 0.0015114
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004282, 0.0004261
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0031595, 0.0031441
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004181, 0.0004161
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0023497, 0.0023612
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006528, 0.0006560
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0005926, 0.0005955
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0022113, 0.0022222
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017295, 0.0017211
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001485, 0.0001492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004187, upper bound: 0.0004302
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004187, upper bound: 0.0004303
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0017190, 0.0017031
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004847, 0.0004802
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0035759, 0.0035428
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004732, 0.0004688
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026477, 0.0026724
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007356, 0.0007425
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006677, 0.0006739
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024918, 0.0025150
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019575, 0.0019393
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001673, 0.0001689

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003248, upper bound: 0.0003243
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003249, upper bound: 0.0003243
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0017417, 0.0017043
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004910, 0.0004805
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0036230, 0.0035454
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004794, 0.0004692
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026496, 0.0027076
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007361, 0.0007523
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006682, 0.0006828
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024936, 0.0025482
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019832, 0.0019407
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001674, 0.0001711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003248, upper bound: 0.0003243
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003249, upper bound: 0.0003243
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0017044, 0.0017726
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004805, 0.0004998
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0035456, 0.0036874
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004692, 0.0004880
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0027557, 0.0026497
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007656, 0.0007362
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006950, 0.0006682
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0025935, 0.0024937
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019409, 0.0020185
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001741, 0.0001674

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003924, upper bound: 0.0003838
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003907, upper bound: 0.0003863
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0017174, 0.0017594
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004842, 0.0004960
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0035725, 0.0036598
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004728, 0.0004843
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0027351, 0.0026698
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007599, 0.0007418
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006898, 0.0006733
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0025741, 0.0025126
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019556, 0.0020034
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001728, 0.0001687

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003880, upper bound: 0.0004013
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003881, upper bound: 0.0004023
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0017063, 0.0016970
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004811, 0.0004784
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0035495, 0.0035301
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004697, 0.0004671
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026382, 0.0026527
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007330, 0.0007370
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006653, 0.0006690
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024828, 0.0024964
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019430, 0.0019324
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001667, 0.0001676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004478, upper bound: 0.0004546
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004478, upper bound: 0.0004547
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0017085, 0.0016959
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004817, 0.0004781
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0035540, 0.0035278
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004703, 0.0004669
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026365, 0.0026560
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007325, 0.0007379
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006649, 0.0006698
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024812, 0.0024996
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019454, 0.0019311
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001666, 0.0001678

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004570, upper bound: 0.0004671
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004570, upper bound: 0.0004672
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0017400, 0.0017155
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004906, 0.0004837
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0036195, 0.0035686
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004790, 0.0004722
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026669, 0.0027050
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007410, 0.0007515
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006726, 0.0006822
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0025099, 0.0025457
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019813, 0.0019534
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001685, 0.0001709

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004694, upper bound: 0.0004515
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004535, upper bound: 0.0004671
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0017434, 0.0017114
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004915, 0.0004825
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0036266, 0.0035600
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004799, 0.0004711
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026605, 0.0027103
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007392, 0.0007530
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006709, 0.0006835
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0025038, 0.0025507
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019852, 0.0019487
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001681, 0.0001713

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004612, upper bound: 0.0004578
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004494, upper bound: 0.0004767
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0017217, 0.0016979
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004854, 0.0004787
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0035815, 0.0035319
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004740, 0.0004674
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026395, 0.0026766
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007333, 0.0007436
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006657, 0.0006750
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024841, 0.0025190
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019605, 0.0019334
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001668, 0.0001691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004472, upper bound: 0.0004528
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004383, upper bound: 0.0004620
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0017412, 0.0016966
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004909, 0.0004783
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0036221, 0.0035294
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004793, 0.0004671
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026376, 0.0027070
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007328, 0.0007521
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006652, 0.0006827
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024823, 0.0025476
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019828, 0.0019320
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001667, 0.0001711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004200, upper bound: 0.0004411
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004240, upper bound: 0.0004381
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0017364, 0.0017019
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004896, 0.0004798
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0036120, 0.0035403
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004780, 0.0004685
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026458, 0.0026994
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007351, 0.0007500
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006672, 0.0006808
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024900, 0.0025404
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019772, 0.0019379
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001672, 0.0001706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004479, upper bound: 0.0004444
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004496, upper bound: 0.0004414
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0017463, 0.0016914
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004924, 0.0004769
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0036327, 0.0035185
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004807, 0.0004656
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026295, 0.0027149
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007306, 0.0007543
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006631, 0.0006846
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024747, 0.0025550
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019885, 0.0019261
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001662, 0.0001716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004517, upper bound: 0.0004640
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004475, upper bound: 0.0004819
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015153, 0.0015274
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004272, 0.0004306
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0031521, 0.0031774
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004171, 0.0004205
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0023746, 0.0023557
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006597, 0.0006545
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0005988, 0.0005941
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0022347, 0.0022170
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017255, 0.0017393
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001501, 0.0001489

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0003076, upper bound: 0.0003052
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0003070, upper bound: 0.0003062
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015173, 0.0015251
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004278, 0.0004300
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0031562, 0.0031726
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004177, 0.0004198
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0023710, 0.0023588
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006587, 0.0006553
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0005979, 0.0005948
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0022314, 0.0022199
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017277, 0.0017367
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001498, 0.0001491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0003053, upper bound: 0.0003024
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0003011, upper bound: 0.0003080
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015367, 0.0015478
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004333, 0.0004364
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0031967, 0.0032198
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004230, 0.0004261
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0024063, 0.0023890
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006685, 0.0006637
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006068, 0.0006025
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0022646, 0.0022483
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017499, 0.0017625
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001521, 0.0001510

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0003095, upper bound: 0.0002958
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0003043, upper bound: 0.0002998
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015400, 0.0015447
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004342, 0.0004355
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0032035, 0.0032133
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004239, 0.0004252
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0024014, 0.0023941
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006672, 0.0006652
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006056, 0.0006038
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0022600, 0.0022531
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017536, 0.0017590
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001518, 0.0001513

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002986, upper bound: 0.0003046
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002951, upper bound: 0.0003101
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0014764, 0.0014933
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004162, 0.0004210
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0030712, 0.0031065
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004064, 0.0004111
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0023216, 0.0022952
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006450, 0.0006377
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0005855, 0.0005788
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0021849, 0.0021600
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0016812, 0.0017005
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001467, 0.0001450

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002326, upper bound: 0.0002239
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002326, upper bound: 0.0002239
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0014788, 0.0014908
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004169, 0.0004203
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0030762, 0.0031012
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004071, 0.0004104
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0023176, 0.0022990
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006439, 0.0006387
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0005845, 0.0005798
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0021812, 0.0021636
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0016839, 0.0016976
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001465, 0.0001453

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004180, upper bound: 0.0004105
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004174, upper bound: 0.0004105
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015002, 0.0014991
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004230, 0.0004227
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0031208, 0.0031184
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004130, 0.0004127
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0023305, 0.0023323
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006475, 0.0006480
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0005877, 0.0005882
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0021933, 0.0021949
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017083, 0.0017070
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001473, 0.0001474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004570, upper bound: 0.0004519
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004566, upper bound: 0.0004520
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015013, 0.0014983
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004233, 0.0004224
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0031229, 0.0031168
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004133, 0.0004125
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0023293, 0.0023339
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006471, 0.0006484
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0005874, 0.0005886
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0021921, 0.0021964
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017095, 0.0017061
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001472, 0.0001475

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004415, upper bound: 0.0004448
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004414, upper bound: 0.0004453
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0014864, 0.0014822
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004191, 0.0004179
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0030920, 0.0030832
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004092, 0.0004080
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0023042, 0.0023108
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006402, 0.0006420
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0005811, 0.0005827
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0021685, 0.0021747
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0016926, 0.0016878
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001456, 0.0001460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003653, upper bound: 0.0003697
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003653, upper bound: 0.0003690
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0014945, 0.0014748
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004214, 0.0004158
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0031089, 0.0030679
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004114, 0.0004060
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0022928, 0.0023234
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006370, 0.0006455
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0005782, 0.0005859
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0021577, 0.0021866
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017018, 0.0016794
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001449, 0.0001468

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003680, upper bound: 0.0003609
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003537, upper bound: 0.0003766
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0014972, 0.0014992
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004221, 0.0004227
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0031145, 0.0031187
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004121, 0.0004127
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0023307, 0.0023276
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006475, 0.0006467
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0005878, 0.0005870
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0021935, 0.0021905
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017049, 0.0017072
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001473, 0.0001471

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004576, upper bound: 0.0004139
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004393, upper bound: 0.0004260
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0014998, 0.0014976
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004228, 0.0004222
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0031198, 0.0031152
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004129, 0.0004123
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0023281, 0.0023316
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006468, 0.0006478
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0005871, 0.0005880
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0021910, 0.0021943
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017078, 0.0017053
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001471, 0.0001473

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004503, upper bound: 0.0004234
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004461, upper bound: 0.0004246
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015418, 0.0015233
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004347, 0.0004295
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0032072, 0.0031687
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004244, 0.0004193
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0023681, 0.0023969
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006579, 0.0006659
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0005972, 0.0006045
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0022287, 0.0022557
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017556, 0.0017346
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001496, 0.0001515

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003312, upper bound: 0.0003217
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003253, upper bound: 0.0003284
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015460, 0.0015181
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004359, 0.0004280
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0032160, 0.0031580
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004256, 0.0004179
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0023601, 0.0024034
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006557, 0.0006677
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0005952, 0.0006061
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0022211, 0.0022619
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017604, 0.0017287
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001491, 0.0001519

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003336, upper bound: 0.0003311
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003334, upper bound: 0.0003312
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015179, 0.0015053
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004280, 0.0004244
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0031576, 0.0031313
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004179, 0.0004144
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0023401, 0.0023598
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006502, 0.0006556
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0005901, 0.0005951
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0022023, 0.0022209
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017285, 0.0017141
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001479, 0.0001491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003223, upper bound: 0.0003223
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003194, upper bound: 0.0003248
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015223, 0.0015005
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004292, 0.0004230
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0031667, 0.0031213
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004191, 0.0004131
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0023327, 0.0023666
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006481, 0.0006575
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0005883, 0.0005968
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0021953, 0.0022273
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017335, 0.0017086
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001474, 0.0001496

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003201, upper bound: 0.0003102
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003120, upper bound: 0.0003226
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015037, 0.0014804
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004240, 0.0004174
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0031280, 0.0030796
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004139, 0.0004075
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0023015, 0.0023377
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006394, 0.0006495
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0005804, 0.0005895
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0021659, 0.0022000
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017123, 0.0016858
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001454, 0.0001477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004155, upper bound: 0.0004308
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004192, upper bound: 0.0004315
time: 1.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015252, 0.0014818
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004300, 0.0004178
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0031727, 0.0030823
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004199, 0.0004079
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0023036, 0.0023711
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006400, 0.0006587
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0005809, 0.0005979
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0021679, 0.0022314
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017367, 0.0016873
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001456, 0.0001498

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004154, upper bound: 0.0004307
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004192, upper bound: 0.0004316
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0014800, 0.0015459
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004173, 0.0004358
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0030788, 0.0032157
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004074, 0.0004255
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0024032, 0.0023009
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006677, 0.0006393
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006061, 0.0005802
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0022617, 0.0021654
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0016853, 0.0017603
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001519, 0.0001454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003987, upper bound: 0.0003781
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003987, upper bound: 0.0003781
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0014810, 0.0015451
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004176, 0.0004356
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0030809, 0.0032141
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004077, 0.0004253
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0024020, 0.0023024
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006673, 0.0006397
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006057, 0.0005806
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0022605, 0.0021669
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0016865, 0.0017594
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001518, 0.0001455

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003948, upper bound: 0.0003796
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003948, upper bound: 0.0003796
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0014215, 0.0014746
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004008, 0.0004158
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0029570, 0.0030675
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0003913, 0.0004059
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0022925, 0.0022098
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006369, 0.0006140
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0005781, 0.0005573
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0021575, 0.0020797
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0016186, 0.0016792
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001449, 0.0001396

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004372, upper bound: 0.0004330
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004373, upper bound: 0.0004330
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0014295, 0.0014680
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004030, 0.0004139
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0029736, 0.0030538
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0003935, 0.0004041
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0022822, 0.0022223
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006341, 0.0006174
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0005755, 0.0005604
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0021478, 0.0020914
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0016277, 0.0016717
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001442, 0.0001404

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004422, upper bound: 0.0004435
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004413, upper bound: 0.0004455
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015005, 0.0015469
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004230, 0.0004361
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0031214, 0.0032180
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004131, 0.0004258
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0024049, 0.0023327
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006682, 0.0006481
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006065, 0.0005883
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0022633, 0.0021953
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017086, 0.0017615
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001520, 0.0001474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 205

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003343, upper bound: 0.0003320
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003300, upper bound: 0.0003357
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015016, 0.0015462
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004234, 0.0004359
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0031236, 0.0032163
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004134, 0.0004256
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0024037, 0.0023344
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006678, 0.0006486
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006062, 0.0005887
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0022621, 0.0021969
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017099, 0.0017606
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001519, 0.0001475

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003356, upper bound: 0.0003288
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003259, upper bound: 0.0003425
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0014420, 0.0014848
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004066, 0.0004186
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0029997, 0.0030887
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0003970, 0.0004087
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0023083, 0.0022418
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006413, 0.0006228
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0005821, 0.0005653
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0021724, 0.0021097
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0016420, 0.0016908
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001459, 0.0001417

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0003040, upper bound: 0.0003088
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0003040, upper bound: 0.0003088
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0014482, 0.0014785
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004083, 0.0004168
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0030125, 0.0030755
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0003987, 0.0004070
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0022984, 0.0022513
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006386, 0.0006255
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0005796, 0.0005678
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0021631, 0.0021188
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0016490, 0.0016835
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001452, 0.0001423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003259, upper bound: 0.0003301
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003258, upper bound: 0.0003302
time: 0.97 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.04 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004532, upper bound: 0.0004250
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004379, upper bound: 0.0004307
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004500, upper bound: 0.0004245
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004388, upper bound: 0.0004357
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004608, upper bound: 0.0004428
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004505, upper bound: 0.0004490
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004551, upper bound: 0.0004439
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004378, upper bound: 0.0004440
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003912, upper bound: 0.0003858
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003912, upper bound: 0.0003858
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003963, upper bound: 0.0003967
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003963, upper bound: 0.0003967
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004266, upper bound: 0.0004164
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004266, upper bound: 0.0004164
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004164, upper bound: 0.0004315
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004192, upper bound: 0.0004301
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004471, upper bound: 0.0004302
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004461, upper bound: 0.0004302
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004338, upper bound: 0.0004391
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004290, upper bound: 0.0004425
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004402, upper bound: 0.0004289
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004277, upper bound: 0.0004383
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004345, upper bound: 0.0004377
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004282, upper bound: 0.0004379
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004352, upper bound: 0.0004146
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004207, upper bound: 0.0004254
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0001882, upper bound: 0.0001892
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0001882, upper bound: 0.0001892
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003109, upper bound: 0.0003129
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003109, upper bound: 0.0003117
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003253, upper bound: 0.0003246
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003236, upper bound: 0.0003275
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004363, upper bound: 0.0004248
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004363, upper bound: 0.0004248
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004377, upper bound: 0.0004443
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004471, upper bound: 0.0004443
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003795, upper bound: 0.0003936
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003795, upper bound: 0.0003936
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004187, upper bound: 0.0004302
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004187, upper bound: 0.0004303
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003248, upper bound: 0.0003243
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003249, upper bound: 0.0003243
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003248, upper bound: 0.0003243
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003249, upper bound: 0.0003243
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003924, upper bound: 0.0003838
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003907, upper bound: 0.0003863
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003880, upper bound: 0.0004013
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003881, upper bound: 0.0004023
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004478, upper bound: 0.0004546
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004478, upper bound: 0.0004547
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004570, upper bound: 0.0004671
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004570, upper bound: 0.0004672
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004694, upper bound: 0.0004515
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004535, upper bound: 0.0004671
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004612, upper bound: 0.0004578
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004494, upper bound: 0.0004767
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004472, upper bound: 0.0004528
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004383, upper bound: 0.0004620
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004200, upper bound: 0.0004411
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004240, upper bound: 0.0004381
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004479, upper bound: 0.0004444
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004496, upper bound: 0.0004414
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004517, upper bound: 0.0004640
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004475, upper bound: 0.0004819
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003076, upper bound: 0.0003052
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003070, upper bound: 0.0003062
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003053, upper bound: 0.0003024
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003011, upper bound: 0.0003080
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003095, upper bound: 0.0002958
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003043, upper bound: 0.0002998
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0002986, upper bound: 0.0003046
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0002951, upper bound: 0.0003101
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0002326, upper bound: 0.0002239
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0002326, upper bound: 0.0002239
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004180, upper bound: 0.0004105
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004174, upper bound: 0.0004105
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004570, upper bound: 0.0004519
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004566, upper bound: 0.0004520
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004415, upper bound: 0.0004448
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004414, upper bound: 0.0004453
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003653, upper bound: 0.0003697
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003653, upper bound: 0.0003690
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003680, upper bound: 0.0003609
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003537, upper bound: 0.0003766
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004576, upper bound: 0.0004139
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004393, upper bound: 0.0004260
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004503, upper bound: 0.0004234
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004461, upper bound: 0.0004246
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003312, upper bound: 0.0003217
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003253, upper bound: 0.0003284
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003336, upper bound: 0.0003311
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003334, upper bound: 0.0003312
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003223, upper bound: 0.0003223
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003194, upper bound: 0.0003248
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003201, upper bound: 0.0003102
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003120, upper bound: 0.0003226
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004155, upper bound: 0.0004308
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004192, upper bound: 0.0004315
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004154, upper bound: 0.0004307
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004192, upper bound: 0.0004316
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003987, upper bound: 0.0003781
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003987, upper bound: 0.0003781
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003948, upper bound: 0.0003796
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003948, upper bound: 0.0003796
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004372, upper bound: 0.0004330
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004373, upper bound: 0.0004330
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004422, upper bound: 0.0004435
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0004413, upper bound: 0.0004455
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003343, upper bound: 0.0003320
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003300, upper bound: 0.0003357
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003356, upper bound: 0.0003288
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003259, upper bound: 0.0003425
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003040, upper bound: 0.0003088
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003040, upper bound: 0.0003088
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003259, upper bound: 0.0003301
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 5, lower bound: -0.0003258, upper bound: 0.0003302

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016008, 0.0016551
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004513, 0.0004666
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0033301, 0.0034430
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004407, 0.0004556
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025731, 0.0024887
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007149, 0.0006914
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006489, 0.0006276
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024216, 0.0023421
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018229, 0.0018847
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001626, 0.0001573

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004482, upper bound: 0.0004160
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004429, upper bound: 0.0004199
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016136, 0.0016396
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004549, 0.0004623
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0033567, 0.0034106
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004442, 0.0004513
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025489, 0.0025086
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007082, 0.0006970
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006428, 0.0006326
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0023988, 0.0023609
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018375, 0.0018670
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001611, 0.0001585

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004356, upper bound: 0.0004183
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004281, upper bound: 0.0004285
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016050, 0.0016492
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004525, 0.0004650
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0033388, 0.0034306
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004418, 0.0004540
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025638, 0.0024952
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007123, 0.0006932
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006466, 0.0006293
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024128, 0.0023483
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018277, 0.0018779
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001620, 0.0001577

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004498, upper bound: 0.0004237
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004477, upper bound: 0.0004243
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016147, 0.0016398
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004552, 0.0004623
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0033589, 0.0034110
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004445, 0.0004514
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025492, 0.0025102
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007082, 0.0006974
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006429, 0.0006330
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0023991, 0.0023624
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018387, 0.0018672
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001611, 0.0001586

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004166, upper bound: 0.0004162
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004192, upper bound: 0.0004143
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016225, 0.0016335
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004574, 0.0004605
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0033751, 0.0033980
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004466, 0.0004497
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025394, 0.0025223
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007055, 0.0007008
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006404, 0.0006361
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0023899, 0.0023738
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018475, 0.0018601
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001605, 0.0001594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004571, upper bound: 0.0004265
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004415, upper bound: 0.0004389
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016260, 0.0016292
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004584, 0.0004593
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0033824, 0.0033891
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004476, 0.0004485
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025328, 0.0025278
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007037, 0.0007023
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006387, 0.0006375
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0023836, 0.0023790
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018515, 0.0018552
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001601, 0.0001597

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004386, upper bound: 0.0004358
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004357, upper bound: 0.0004361
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016426, 0.0016332
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004631, 0.0004605
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0034170, 0.0033973
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004522, 0.0004496
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025389, 0.0025537
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007054, 0.0007095
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006403, 0.0006440
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0023894, 0.0024033
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018705, 0.0018597
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001604, 0.0001614

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004485, upper bound: 0.0004368
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004477, upper bound: 0.0004371
time: 1.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016459, 0.0016277
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004640, 0.0004589
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0034239, 0.0033860
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004531, 0.0004481
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025305, 0.0025588
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007031, 0.0007109
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006382, 0.0006453
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0023815, 0.0024081
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018742, 0.0018535
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001599, 0.0001617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004512, upper bound: 0.0004285
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004254, upper bound: 0.0004400
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015981, 0.0016235
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004506, 0.0004577
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0033243, 0.0033773
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004399, 0.0004469
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025240, 0.0024844
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007012, 0.0006902
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006365, 0.0006265
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0023753, 0.0023381
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018197, 0.0018487
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001595, 0.0001570

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003529, upper bound: 0.0003465
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003529, upper bound: 0.0003465
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016194, 0.0016253
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004566, 0.0004582
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0033686, 0.0033810
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004458, 0.0004474
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025267, 0.0025175
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007020, 0.0006994
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006372, 0.0006349
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0023779, 0.0023692
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018440, 0.0018507
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001597, 0.0001591

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003869, upper bound: 0.0003730
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003786, upper bound: 0.0003813
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016057, 0.0016192
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004527, 0.0004565
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0033402, 0.0033683
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004420, 0.0004457
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025173, 0.0024963
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006994, 0.0006935
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006348, 0.0006295
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0023690, 0.0023493
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018284, 0.0018438
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001591, 0.0001577

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003512, upper bound: 0.0003521
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003512, upper bound: 0.0003521
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016255, 0.0016172
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004583, 0.0004559
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0033814, 0.0033640
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004475, 0.0004452
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025141, 0.0025271
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006985, 0.0007021
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006340, 0.0006373
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0023660, 0.0023783
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018510, 0.0018415
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001589, 0.0001597

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003896, upper bound: 0.0003898
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003893, upper bound: 0.0003902
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016445, 0.0016780
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004636, 0.0004731
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0034209, 0.0034906
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004527, 0.0004619
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026086, 0.0025566
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007248, 0.0007103
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006579, 0.0006447
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024550, 0.0024060
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018726, 0.0019107
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001648, 0.0001616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004193, upper bound: 0.0004076
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004154, upper bound: 0.0004078
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016475, 0.0016732
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004645, 0.0004717
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0034271, 0.0034806
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004535, 0.0004606
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026012, 0.0025612
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007227, 0.0007116
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006560, 0.0006459
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024480, 0.0024104
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018760, 0.0019053
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001644, 0.0001619

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004129, upper bound: 0.0004015
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004124, upper bound: 0.0004027
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0014161, 0.0014218
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0003993, 0.0004009
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0029459, 0.0029577
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0003898, 0.0003914
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0022104, 0.0022016
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006141, 0.0006117
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0005574, 0.0005552
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0020802, 0.0020719
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0016126, 0.0016190
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001397, 0.0001391

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003639, upper bound: 0.0003716
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003639, upper bound: 0.0003716
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0014192, 0.0014208
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004001, 0.0004006
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0029521, 0.0029555
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0003907, 0.0003911
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0022088, 0.0022062
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006137, 0.0006130
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0005570, 0.0005564
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0020787, 0.0020763
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0016160, 0.0016179
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001396, 0.0001394

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004189, upper bound: 0.0004282
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004187, upper bound: 0.0004299
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015971, 0.0016323
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004503, 0.0004602
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0033224, 0.0033955
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004397, 0.0004493
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025376, 0.0024829
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007050, 0.0006898
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006399, 0.0006262
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0023882, 0.0023367
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018187, 0.0018587
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001604, 0.0001569

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003955, upper bound: 0.0003844
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003955, upper bound: 0.0003844
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015984, 0.0016306
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004507, 0.0004597
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0033250, 0.0033920
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004400, 0.0004489
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025350, 0.0024849
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007043, 0.0006904
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006393, 0.0006267
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0023857, 0.0023386
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018201, 0.0018568
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001602, 0.0001570

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004386, upper bound: 0.0004163
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004298, upper bound: 0.0004221
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016205, 0.0016379
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004569, 0.0004618
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0033710, 0.0034073
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004461, 0.0004509
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025464, 0.0025192
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007075, 0.0006999
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006422, 0.0006353
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0023964, 0.0023709
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018453, 0.0018651
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001609, 0.0001592

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004261, upper bound: 0.0004286
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004227, upper bound: 0.0004314
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016241, 0.0016339
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004579, 0.0004607
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0033784, 0.0033988
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004471, 0.0004498
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025401, 0.0025248
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007057, 0.0007015
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006406, 0.0006367
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0023905, 0.0023761
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018493, 0.0018605
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001605, 0.0001596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003807, upper bound: 0.0003954
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003807, upper bound: 0.0003954
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016168, 0.0016608
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004558, 0.0004683
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0033633, 0.0034549
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004451, 0.0004572
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025820, 0.0025135
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007173, 0.0006983
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006511, 0.0006339
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024299, 0.0023655
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018411, 0.0018912
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001632, 0.0001588

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004340, upper bound: 0.0004221
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004328, upper bound: 0.0004221
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016305, 0.0016448
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004597, 0.0004637
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0033918, 0.0034215
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004489, 0.0004528
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025570, 0.0025348
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007104, 0.0007043
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006448, 0.0006393
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024065, 0.0023856
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018567, 0.0018730
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001616, 0.0001602

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003820, upper bound: 0.0003881
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003820, upper bound: 0.0003881
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016132, 0.0016266
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004548, 0.0004586
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0033558, 0.0033837
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004441, 0.0004478
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025288, 0.0025079
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007026, 0.0006968
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006377, 0.0006325
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0023799, 0.0023602
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018369, 0.0018523
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001598, 0.0001585

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004342, upper bound: 0.0004353
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004334, upper bound: 0.0004375
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016150, 0.0016250
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004553, 0.0004581
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0033596, 0.0033802
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004446, 0.0004473
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025262, 0.0025107
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007018, 0.0006976
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006371, 0.0006332
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0023774, 0.0023629
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018390, 0.0018503
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001596, 0.0001587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003720, upper bound: 0.0003755
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003720, upper bound: 0.0003755
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016484, 0.0016725
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004647, 0.0004715
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0034290, 0.0034791
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004538, 0.0004604
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026001, 0.0025626
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007224, 0.0007120
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006557, 0.0006463
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024470, 0.0024117
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018770, 0.0019045
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001643, 0.0001619

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004256, upper bound: 0.0004004
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004148, upper bound: 0.0004056
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016614, 0.0016596
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004684, 0.0004679
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0034561, 0.0034522
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004574, 0.0004568
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0025800, 0.0025829
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007168, 0.0007176
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006506, 0.0006514
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024280, 0.0024308
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018919, 0.0018897
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001630, 0.0001632

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004116, upper bound: 0.0004079
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004055, upper bound: 0.0004169
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016399, 0.0016883
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004624, 0.0004760
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0034113, 0.0035121
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004514, 0.0004648
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026247, 0.0025494
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007292, 0.0007083
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006619, 0.0006429
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024702, 0.0023993
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018674, 0.0019225
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001659, 0.0001611

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003203, upper bound: 0.0003147
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003112, upper bound: 0.0003193
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016407, 0.0016874
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004626, 0.0004757
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0034131, 0.0035102
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004517, 0.0004645
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026233, 0.0025507
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007288, 0.0007087
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006616, 0.0006433
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024688, 0.0024005
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018683, 0.0019215
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001658, 0.0001612

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0003151, upper bound: 0.0003119
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003095, upper bound: 0.0003195
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016738, 0.0017020
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004719, 0.0004799
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0034819, 0.0035405
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004608, 0.0004685
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026459, 0.0026021
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007351, 0.0007229
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006673, 0.0006562
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024901, 0.0024489
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019060, 0.0019381
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001672, 0.0001644

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004287, upper bound: 0.0004154
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004260, upper bound: 0.0004169
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016770, 0.0016976
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004728, 0.0004786
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0034886, 0.0035314
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004617, 0.0004673
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026392, 0.0026071
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007332, 0.0007243
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006656, 0.0006575
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024838, 0.0024536
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019097, 0.0019331
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001668, 0.0001648

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004312, upper bound: 0.0004152
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004286, upper bound: 0.0004198
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016667, 0.0016793
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004699, 0.0004735
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0034671, 0.0034933
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004588, 0.0004623
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026106, 0.0025911
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007253, 0.0007199
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006584, 0.0006534
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024569, 0.0024385
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018979, 0.0019122
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001650, 0.0001637

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004449, upper bound: 0.0004351
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004323, upper bound: 0.0004422
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016686, 0.0016777
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004704, 0.0004730
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0034709, 0.0034900
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004593, 0.0004618
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026082, 0.0025940
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007246, 0.0007207
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006577, 0.0006542
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024546, 0.0024412
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019000, 0.0019104
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001648, 0.0001639

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004449, upper bound: 0.0004351
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004327, upper bound: 0.0004422
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0014835, 0.0014798
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004182, 0.0004172
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0030859, 0.0030783
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004084, 0.0004074
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0023005, 0.0023062
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006392, 0.0006407
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0005802, 0.0005816
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0021651, 0.0021704
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0016892, 0.0016851
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001454, 0.0001457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0003010, upper bound: 0.0003083
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0003010, upper bound: 0.0003083
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015126, 0.0014810
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004265, 0.0004176
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0031465, 0.0030809
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004164, 0.0004077
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0023025, 0.0023515
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006397, 0.0006533
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0005806, 0.0005930
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0021669, 0.0022130
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017224, 0.0016865
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001455, 0.0001486

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 205

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003738, upper bound: 0.0003834
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003674, upper bound: 0.0003873
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015027, 0.0014978
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004237, 0.0004223
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0031259, 0.0031157
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004137, 0.0004123
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0023285, 0.0023361
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006469, 0.0006490
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0005872, 0.0005891
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0021914, 0.0021985
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017111, 0.0017056
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001471, 0.0001476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004105, upper bound: 0.0004197
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004096, upper bound: 0.0004226
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0015060, 0.0014953
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004246, 0.0004216
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0031329, 0.0031105
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004146, 0.0004116
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0023246, 0.0023413
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0006458, 0.0006505
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0005862, 0.0005904
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0021877, 0.0022034
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0017149, 0.0017027
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001469, 0.0001480

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003472, upper bound: 0.0003559
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003472, upper bound: 0.0003559
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016909, 0.0016761
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004767, 0.0004725
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0035174, 0.0034866
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004655, 0.0004614
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026057, 0.0026287
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007239, 0.0007303
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006571, 0.0006629
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024522, 0.0024739
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019254, 0.0019086
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001647, 0.0001661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0003069, upper bound: 0.0003062
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0003069, upper bound: 0.0003063
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016920, 0.0016754
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004770, 0.0004724
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0035198, 0.0034853
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004658, 0.0004612
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026047, 0.0026305
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007237, 0.0007308
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006569, 0.0006634
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024513, 0.0024756
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019267, 0.0019078
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001646, 0.0001662

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002787, upper bound: 0.0002784
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002787, upper bound: 0.0002784
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0017141, 0.0016777
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004833, 0.0004730
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0035656, 0.0034900
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004718, 0.0004618
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026082, 0.0026647
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007246, 0.0007403
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006578, 0.0006720
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024546, 0.0025078
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019518, 0.0019104
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001648, 0.0001684

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003204, upper bound: 0.0003115
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003116, upper bound: 0.0003197
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0017152, 0.0016767
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004836, 0.0004727
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0035680, 0.0034878
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004722, 0.0004616
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026066, 0.0026665
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007242, 0.0007408
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006573, 0.0006725
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024531, 0.0025095
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019531, 0.0019092
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001647, 0.0001685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003170, upper bound: 0.0003140
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003139, upper bound: 0.0003163
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016477, 0.0017209
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004646, 0.0004852
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0034276, 0.0035798
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004536, 0.0004737
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026753, 0.0025616
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007433, 0.0007117
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006747, 0.0006460
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0025178, 0.0024107
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018763, 0.0019596
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001691, 0.0001619

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 205

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003865, upper bound: 0.0003735
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003778, upper bound: 0.0003774
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016477, 0.0017159
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004645, 0.0004838
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0034275, 0.0035694
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004536, 0.0004724
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026676, 0.0025615
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007411, 0.0007117
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006727, 0.0006460
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0025105, 0.0024106
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0018762, 0.0019539
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001686, 0.0001619

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 205

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003848, upper bound: 0.0003742
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003773, upper bound: 0.0003803
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016897, 0.0017331
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004764, 0.0004886
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0035149, 0.0036052
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004651, 0.0004771
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026943, 0.0026268
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007486, 0.0007298
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006795, 0.0006624
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0025357, 0.0024721
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019240, 0.0019735
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001703, 0.0001660

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 205

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003821, upper bound: 0.0003904
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003754, upper bound: 0.0003951
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016920, 0.0017321
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004770, 0.0004883
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0035197, 0.0036030
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004658, 0.0004768
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026927, 0.0026304
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007481, 0.0007308
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006791, 0.0006633
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0025341, 0.0024755
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019267, 0.0019723
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001702, 0.0001662

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003755, upper bound: 0.0003875
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003725, upper bound: 0.0003893
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016863, 0.0016787
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004754, 0.0004733
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0035079, 0.0034921
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004642, 0.0004621
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026097, 0.0026216
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007251, 0.0007284
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006581, 0.0006611
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024561, 0.0024672
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019202, 0.0019116
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001649, 0.0001657

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004451, upper bound: 0.0004335
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004284, upper bound: 0.0004519
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0017063, 0.0016770
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004811, 0.0004728
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0035495, 0.0034885
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004697, 0.0004616
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026071, 0.0026527
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007243, 0.0007370
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006575, 0.0006690
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024536, 0.0024964
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019430, 0.0019096
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001648, 0.0001676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004353, upper bound: 0.0004420
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004352, upper bound: 0.0004420
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0089628, -0.0060465, -0.0089628, -0.0060465, -0.0016978, 0.0016900
1: -0.0054656, -0.0046434, -0.0054656, -0.0046434, -0.0004787, 0.0004765
2: -0.0017665, 0.0042999, -0.0017665, 0.0042999, -0.0035318, 0.0035156
3: 0.0013935, 0.0021963, 0.0013935, 0.0021963, -0.0004674, 0.0004652
4: 0.0028784, 0.0074120, 0.0028784, 0.0074120, -0.0026273, 0.0026395
5: 0.9963060, 0.9975656, 0.9963060, 0.9975656, -0.0007300, 0.0007333
6: 0.0045306, 0.0056739, 0.0045306, 0.0056739, -0.0006626, 0.0006656
7: -0.0064743, -0.0022076, -0.0064743, -0.0022076, -0.0024726, 0.0024840
8: -0.0074747, -0.0041539, -0.0074747, -0.0041539, -0.0019333, 0.0019244
9: -0.0036514, -0.0033649, -0.0036514, -0.0033649, -0.0001660, 0.0001668

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 3.11 + 597.20 = 600.31 seconds
