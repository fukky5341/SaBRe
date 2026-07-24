## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.1285293


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087)
1: (-0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209)
2: (-0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101)
3: (-0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673)
4: (-0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748)
5: (-0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771)
6: (-0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318)
7: (0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096)
8: (-0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122)
9: (-0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.67 + 2.00 = 3.68 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.85 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.87 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.87
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.87
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.77 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.83 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.33 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.33
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.33
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.33
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.33
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.05 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.90 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.90
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.90
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.90
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.90
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.90
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.90
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.90
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.90
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.05 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.86 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.88 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.50 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.92 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.56 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.25 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.25 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.25 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.10 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 4.16 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1875160, upper bound: 0.1875160
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0857932, 0.0676155, -0.0857932, 0.0676155, -0.1534087, 0.1534087
1: -0.0795514, 0.0725696, -0.0795514, 0.0725696, -0.1521209, 0.1521209
2: -0.0885612, 0.1796490, -0.0885612, 0.1796490, -0.2682101, 0.2682101
3: -0.0631400, 0.0823273, -0.0631400, 0.0823273, -0.1454673, 0.1454673
4: -0.0922399, 0.0977349, -0.0922399, 0.0977349, -0.1899748, 0.1899748
5: -0.0717339, 0.1047432, -0.0717339, 0.1047432, -0.1764771, 0.1764771
6: -0.1386945, 0.1184372, -0.1386945, 0.1184372, -0.2571318, 0.2571318
7: 0.7889843, 1.0291939, 0.7889843, 1.0291939, -0.2402096, 0.2402096
8: -0.1278072, 0.1377050, -0.1278072, 0.1377050, -0.2655122, 0.2655122
9: -0.1074916, 0.1566667, -0.1074916, 0.1566667, -0.2641582, 0.2641582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.81 seconds

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 3.68 + 596.99 = 600.67 seconds
