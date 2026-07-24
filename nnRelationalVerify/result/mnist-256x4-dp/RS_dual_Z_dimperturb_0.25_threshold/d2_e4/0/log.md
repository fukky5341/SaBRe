## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00070371


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0058082, 0.0064537, 0.0058082, 0.0064537, -0.0006454, 0.0006454)
1: (-0.0009467, 0.0005822, -0.0009467, 0.0005822, -0.0015289, 0.0015289)
2: (0.0122316, 0.0226647, 0.0122316, 0.0226647, -0.0104331, 0.0104331)
3: (-0.0045056, -0.0036050, -0.0045056, -0.0036050, -0.0009006, 0.0009006)
4: (-0.0005242, 0.0038452, -0.0005242, 0.0038452, -0.0043694, 0.0043694)
5: (-0.0010937, -0.0002131, -0.0010937, -0.0002131, -0.0008806, 0.0008806)
6: (0.9904653, 0.9922591, 0.9904653, 0.9922591, -0.0017938, 0.0017938)
7: (-0.0144471, -0.0064224, -0.0144471, -0.0064224, -0.0058067, 0.0058067)
8: (-0.0052060, -0.0010237, -0.0052060, -0.0010237, -0.0041823, 0.0041823)
9: (-0.0052859, -0.0001143, -0.0052859, -0.0001143, -0.0051716, 0.0051716)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.58 + 1.92 = 3.50 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0010878, upper bound: 0.0010878

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010817, upper bound: 0.0010840
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010840, upper bound: 0.0010817
time: 1.08 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.20 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.20
Output dim: 6, lower bound: -0.0010817, upper bound: 0.0010840
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.20
Output dim: 6, lower bound: -0.0010840, upper bound: 0.0010817

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058082, 0.0064537, 0.0058082, 0.0064537, -0.0006454, 0.0006454
1: -0.0009467, 0.0005822, -0.0009467, 0.0005822, -0.0015289, 0.0015289
2: 0.0122316, 0.0226647, 0.0122316, 0.0226647, -0.0104331, 0.0104331
3: -0.0045056, -0.0036050, -0.0045056, -0.0036050, -0.0009006, 0.0009006
4: -0.0005242, 0.0038452, -0.0005242, 0.0038452, -0.0043694, 0.0043694
5: -0.0010937, -0.0002131, -0.0010937, -0.0002131, -0.0008806, 0.0008806
6: 0.9904653, 0.9922591, 0.9904653, 0.9922591, -0.0017938, 0.0017938
7: -0.0144471, -0.0064224, -0.0144471, -0.0064224, -0.0057720, 0.0057787
8: -0.0052060, -0.0010237, -0.0052060, -0.0010237, -0.0041823, 0.0041823
9: -0.0052859, -0.0001143, -0.0052859, -0.0001143, -0.0051716, 0.0051716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010311, upper bound: 0.0010336
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010311, upper bound: 0.0010336
time: 0.97 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058082, 0.0064537, 0.0058082, 0.0064537, -0.0006454, 0.0006454
1: -0.0009467, 0.0005822, -0.0009467, 0.0005822, -0.0015289, 0.0015289
2: 0.0122316, 0.0226647, 0.0122316, 0.0226647, -0.0104331, 0.0104331
3: -0.0045056, -0.0036050, -0.0045056, -0.0036050, -0.0009006, 0.0009006
4: -0.0005242, 0.0038452, -0.0005242, 0.0038452, -0.0043694, 0.0043694
5: -0.0010937, -0.0002131, -0.0010937, -0.0002131, -0.0008806, 0.0008806
6: 0.9904653, 0.9922591, 0.9904653, 0.9922591, -0.0017938, 0.0017938
7: -0.0144471, -0.0064224, -0.0144471, -0.0064224, -0.0057787, 0.0057720
8: -0.0052060, -0.0010237, -0.0052060, -0.0010237, -0.0041823, 0.0041823
9: -0.0052859, -0.0001143, -0.0052859, -0.0001143, -0.0051716, 0.0051716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010336, upper bound: 0.0010310
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010336, upper bound: 0.0010311
time: 1.01 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.52 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.52
Output dim: 6, lower bound: -0.0010311, upper bound: 0.0010336
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.52
Output dim: 6, lower bound: -0.0010311, upper bound: 0.0010336
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.52
Output dim: 6, lower bound: -0.0010336, upper bound: 0.0010310
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.52
Output dim: 6, lower bound: -0.0010336, upper bound: 0.0010311

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058082, 0.0064537, 0.0058082, 0.0064537, -0.0006454, 0.0006454
1: -0.0009467, 0.0005822, -0.0009467, 0.0005822, -0.0015289, 0.0015289
2: 0.0122316, 0.0226647, 0.0122316, 0.0226647, -0.0104331, 0.0104331
3: -0.0045056, -0.0036050, -0.0045056, -0.0036050, -0.0009006, 0.0009006
4: -0.0005242, 0.0038452, -0.0005242, 0.0038452, -0.0043694, 0.0043694
5: -0.0010937, -0.0002131, -0.0010937, -0.0002131, -0.0008806, 0.0008806
6: 0.9904653, 0.9922591, 0.9904653, 0.9922591, -0.0017938, 0.0017938
7: -0.0144471, -0.0064224, -0.0144471, -0.0064224, -0.0056876, 0.0057025
8: -0.0052060, -0.0010237, -0.0052060, -0.0010237, -0.0041823, 0.0041823
9: -0.0052859, -0.0001143, -0.0052859, -0.0001143, -0.0051716, 0.0051716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009312, upper bound: 0.0009320
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009312, upper bound: 0.0009320
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058082, 0.0064537, 0.0058082, 0.0064537, -0.0006454, 0.0006454
1: -0.0009467, 0.0005822, -0.0009467, 0.0005822, -0.0015289, 0.0015289
2: 0.0122316, 0.0226647, 0.0122316, 0.0226647, -0.0104331, 0.0104331
3: -0.0045056, -0.0036050, -0.0045056, -0.0036050, -0.0009006, 0.0009006
4: -0.0005242, 0.0038452, -0.0005242, 0.0038452, -0.0043694, 0.0043694
5: -0.0010937, -0.0002131, -0.0010937, -0.0002131, -0.0008806, 0.0008806
6: 0.9904653, 0.9922591, 0.9904653, 0.9922591, -0.0017938, 0.0017938
7: -0.0144471, -0.0064224, -0.0144471, -0.0064224, -0.0056959, 0.0056922
8: -0.0052060, -0.0010237, -0.0052060, -0.0010237, -0.0041823, 0.0041823
9: -0.0052859, -0.0001143, -0.0052859, -0.0001143, -0.0051716, 0.0051716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009312, upper bound: 0.0009320
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009312, upper bound: 0.0009320
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058082, 0.0064537, 0.0058082, 0.0064537, -0.0006454, 0.0006454
1: -0.0009467, 0.0005822, -0.0009467, 0.0005822, -0.0015289, 0.0015289
2: 0.0122316, 0.0226647, 0.0122316, 0.0226647, -0.0104331, 0.0104331
3: -0.0045056, -0.0036050, -0.0045056, -0.0036050, -0.0009006, 0.0009006
4: -0.0005242, 0.0038452, -0.0005242, 0.0038452, -0.0043694, 0.0043694
5: -0.0010937, -0.0002131, -0.0010937, -0.0002131, -0.0008806, 0.0008806
6: 0.9904653, 0.9922591, 0.9904653, 0.9922591, -0.0017938, 0.0017938
7: -0.0144471, -0.0064224, -0.0144471, -0.0064224, -0.0056922, 0.0056959
8: -0.0052060, -0.0010237, -0.0052060, -0.0010237, -0.0041823, 0.0041823
9: -0.0052859, -0.0001143, -0.0052859, -0.0001143, -0.0051716, 0.0051716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009320, upper bound: 0.0009312
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009320, upper bound: 0.0009312
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058082, 0.0064537, 0.0058082, 0.0064537, -0.0006454, 0.0006454
1: -0.0009467, 0.0005822, -0.0009467, 0.0005822, -0.0015289, 0.0015289
2: 0.0122316, 0.0226647, 0.0122316, 0.0226647, -0.0104331, 0.0104331
3: -0.0045056, -0.0036050, -0.0045056, -0.0036050, -0.0009006, 0.0009006
4: -0.0005242, 0.0038452, -0.0005242, 0.0038452, -0.0043694, 0.0043694
5: -0.0010937, -0.0002131, -0.0010937, -0.0002131, -0.0008806, 0.0008806
6: 0.9904653, 0.9922591, 0.9904653, 0.9922591, -0.0017938, 0.0017938
7: -0.0144471, -0.0064224, -0.0144471, -0.0064224, -0.0057025, 0.0056876
8: -0.0052060, -0.0010237, -0.0052060, -0.0010237, -0.0041823, 0.0041823
9: -0.0052859, -0.0001143, -0.0052859, -0.0001143, -0.0051716, 0.0051716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009320, upper bound: 0.0009312
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009320, upper bound: 0.0009312
time: 1.01 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.65 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.65
Output dim: 6, lower bound: -0.0009312, upper bound: 0.0009320
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.65
Output dim: 6, lower bound: -0.0009312, upper bound: 0.0009320
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.65
Output dim: 6, lower bound: -0.0009312, upper bound: 0.0009320
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.65
Output dim: 6, lower bound: -0.0009312, upper bound: 0.0009320
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.65
Output dim: 6, lower bound: -0.0009320, upper bound: 0.0009312
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.65
Output dim: 6, lower bound: -0.0009320, upper bound: 0.0009312
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.65
Output dim: 6, lower bound: -0.0009320, upper bound: 0.0009312
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.65
Output dim: 6, lower bound: -0.0009320, upper bound: 0.0009312

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058082, 0.0064537, 0.0058082, 0.0064537, -0.0006454, 0.0006454
1: -0.0009467, 0.0005822, -0.0009467, 0.0005822, -0.0015289, 0.0015289
2: 0.0122316, 0.0226647, 0.0122316, 0.0226647, -0.0104331, 0.0104331
3: -0.0045056, -0.0036050, -0.0045056, -0.0036050, -0.0009006, 0.0009006
4: -0.0005242, 0.0038452, -0.0005242, 0.0038452, -0.0043694, 0.0043694
5: -0.0010937, -0.0002131, -0.0010937, -0.0002131, -0.0008806, 0.0008806
6: 0.9904653, 0.9922591, 0.9904653, 0.9922591, -0.0017938, 0.0017938
7: -0.0144471, -0.0064224, -0.0144471, -0.0064224, -0.0056690, 0.0056871
8: -0.0052060, -0.0010237, -0.0052060, -0.0010237, -0.0041823, 0.0041823
9: -0.0052859, -0.0001143, -0.0052859, -0.0001143, -0.0051716, 0.0051716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006776, upper bound: 0.0006791
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006776, upper bound: 0.0006791
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058082, 0.0064537, 0.0058082, 0.0064537, -0.0006454, 0.0006454
1: -0.0009467, 0.0005822, -0.0009467, 0.0005822, -0.0015289, 0.0015289
2: 0.0122316, 0.0226647, 0.0122316, 0.0226647, -0.0104331, 0.0104331
3: -0.0045056, -0.0036050, -0.0045056, -0.0036050, -0.0009006, 0.0009006
4: -0.0005242, 0.0038452, -0.0005242, 0.0038452, -0.0043694, 0.0043694
5: -0.0010937, -0.0002131, -0.0010937, -0.0002131, -0.0008806, 0.0008806
6: 0.9904653, 0.9922591, 0.9904653, 0.9922591, -0.0017938, 0.0017938
7: -0.0144471, -0.0064224, -0.0144471, -0.0064224, -0.0056722, 0.0057025
8: -0.0052060, -0.0010237, -0.0052060, -0.0010237, -0.0041823, 0.0041823
9: -0.0052859, -0.0001143, -0.0052859, -0.0001143, -0.0051716, 0.0051716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006776, upper bound: 0.0006791
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006776, upper bound: 0.0006791
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058082, 0.0064537, 0.0058082, 0.0064537, -0.0006454, 0.0006454
1: -0.0009467, 0.0005822, -0.0009467, 0.0005822, -0.0015289, 0.0015289
2: 0.0122316, 0.0226647, 0.0122316, 0.0226647, -0.0104331, 0.0104331
3: -0.0045056, -0.0036050, -0.0045056, -0.0036050, -0.0009006, 0.0009006
4: -0.0005242, 0.0038452, -0.0005242, 0.0038452, -0.0043694, 0.0043694
5: -0.0010937, -0.0002131, -0.0010937, -0.0002131, -0.0008806, 0.0008806
6: 0.9904653, 0.9922591, 0.9904653, 0.9922591, -0.0017938, 0.0017938
7: -0.0144471, -0.0064224, -0.0144471, -0.0064224, -0.0056773, 0.0056768
8: -0.0052060, -0.0010237, -0.0052060, -0.0010237, -0.0041823, 0.0041823
9: -0.0052859, -0.0001143, -0.0052859, -0.0001143, -0.0051716, 0.0051716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006776, upper bound: 0.0006791
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006776, upper bound: 0.0006791
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058082, 0.0064537, 0.0058082, 0.0064537, -0.0006454, 0.0006454
1: -0.0009467, 0.0005822, -0.0009467, 0.0005822, -0.0015289, 0.0015289
2: 0.0122316, 0.0226647, 0.0122316, 0.0226647, -0.0104331, 0.0104331
3: -0.0045056, -0.0036050, -0.0045056, -0.0036050, -0.0009006, 0.0009006
4: -0.0005242, 0.0038452, -0.0005242, 0.0038452, -0.0043694, 0.0043694
5: -0.0010937, -0.0002131, -0.0010937, -0.0002131, -0.0008806, 0.0008806
6: 0.9904653, 0.9922591, 0.9904653, 0.9922591, -0.0017938, 0.0017938
7: -0.0144471, -0.0064224, -0.0144471, -0.0064224, -0.0056805, 0.0056922
8: -0.0052060, -0.0010237, -0.0052060, -0.0010237, -0.0041823, 0.0041823
9: -0.0052859, -0.0001143, -0.0052859, -0.0001143, -0.0051716, 0.0051716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006776, upper bound: 0.0006791
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006776, upper bound: 0.0006791
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058082, 0.0064537, 0.0058082, 0.0064537, -0.0006454, 0.0006454
1: -0.0009467, 0.0005822, -0.0009467, 0.0005822, -0.0015289, 0.0015289
2: 0.0122316, 0.0226647, 0.0122316, 0.0226647, -0.0104331, 0.0104331
3: -0.0045056, -0.0036050, -0.0045056, -0.0036050, -0.0009006, 0.0009006
4: -0.0005242, 0.0038452, -0.0005242, 0.0038452, -0.0043694, 0.0043694
5: -0.0010937, -0.0002131, -0.0010937, -0.0002131, -0.0008806, 0.0008806
6: 0.9904653, 0.9922591, 0.9904653, 0.9922591, -0.0017938, 0.0017938
7: -0.0144471, -0.0064224, -0.0144471, -0.0064224, -0.0056737, 0.0056805
8: -0.0052060, -0.0010237, -0.0052060, -0.0010237, -0.0041823, 0.0041823
9: -0.0052859, -0.0001143, -0.0052859, -0.0001143, -0.0051716, 0.0051716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006791, upper bound: 0.0006776
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006791, upper bound: 0.0006776
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058082, 0.0064537, 0.0058082, 0.0064537, -0.0006454, 0.0006454
1: -0.0009467, 0.0005822, -0.0009467, 0.0005822, -0.0015289, 0.0015289
2: 0.0122316, 0.0226647, 0.0122316, 0.0226647, -0.0104331, 0.0104331
3: -0.0045056, -0.0036050, -0.0045056, -0.0036050, -0.0009006, 0.0009006
4: -0.0005242, 0.0038452, -0.0005242, 0.0038452, -0.0043694, 0.0043694
5: -0.0010937, -0.0002131, -0.0010937, -0.0002131, -0.0008806, 0.0008806
6: 0.9904653, 0.9922591, 0.9904653, 0.9922591, -0.0017938, 0.0017938
7: -0.0144471, -0.0064224, -0.0144471, -0.0064224, -0.0056768, 0.0056959
8: -0.0052060, -0.0010237, -0.0052060, -0.0010237, -0.0041823, 0.0041823
9: -0.0052859, -0.0001143, -0.0052859, -0.0001143, -0.0051716, 0.0051716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006791, upper bound: 0.0006776
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006791, upper bound: 0.0006776
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058082, 0.0064537, 0.0058082, 0.0064537, -0.0006454, 0.0006454
1: -0.0009467, 0.0005822, -0.0009467, 0.0005822, -0.0015289, 0.0015289
2: 0.0122316, 0.0226647, 0.0122316, 0.0226647, -0.0104331, 0.0104331
3: -0.0045056, -0.0036050, -0.0045056, -0.0036050, -0.0009006, 0.0009006
4: -0.0005242, 0.0038452, -0.0005242, 0.0038452, -0.0043694, 0.0043694
5: -0.0010937, -0.0002131, -0.0010937, -0.0002131, -0.0008806, 0.0008806
6: 0.9904653, 0.9922591, 0.9904653, 0.9922591, -0.0017938, 0.0017938
7: -0.0144471, -0.0064224, -0.0144471, -0.0064224, -0.0056833, 0.0056722
8: -0.0052060, -0.0010237, -0.0052060, -0.0010237, -0.0041823, 0.0041823
9: -0.0052859, -0.0001143, -0.0052859, -0.0001143, -0.0051716, 0.0051716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006791, upper bound: 0.0006776
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006791, upper bound: 0.0006776
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058082, 0.0064537, 0.0058082, 0.0064537, -0.0006454, 0.0006454
1: -0.0009467, 0.0005822, -0.0009467, 0.0005822, -0.0015289, 0.0015289
2: 0.0122316, 0.0226647, 0.0122316, 0.0226647, -0.0104331, 0.0104331
3: -0.0045056, -0.0036050, -0.0045056, -0.0036050, -0.0009006, 0.0009006
4: -0.0005242, 0.0038452, -0.0005242, 0.0038452, -0.0043694, 0.0043694
5: -0.0010937, -0.0002131, -0.0010937, -0.0002131, -0.0008806, 0.0008806
6: 0.9904653, 0.9922591, 0.9904653, 0.9922591, -0.0017938, 0.0017938
7: -0.0144471, -0.0064224, -0.0144471, -0.0064224, -0.0056871, 0.0056876
8: -0.0052060, -0.0010237, -0.0052060, -0.0010237, -0.0041823, 0.0041823
9: -0.0052859, -0.0001143, -0.0052859, -0.0001143, -0.0051716, 0.0051716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006791, upper bound: 0.0006776
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006791, upper bound: 0.0006776
time: 0.72 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.19 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 6, lower bound: -0.0006776, upper bound: 0.0006791
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 6, lower bound: -0.0006776, upper bound: 0.0006791
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 6, lower bound: -0.0006776, upper bound: 0.0006791
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 6, lower bound: -0.0006776, upper bound: 0.0006791
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 6, lower bound: -0.0006776, upper bound: 0.0006791
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 6, lower bound: -0.0006776, upper bound: 0.0006791
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 6, lower bound: -0.0006776, upper bound: 0.0006791
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 6, lower bound: -0.0006776, upper bound: 0.0006791
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 6, lower bound: -0.0006791, upper bound: 0.0006776
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 6, lower bound: -0.0006791, upper bound: 0.0006776
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 6, lower bound: -0.0006791, upper bound: 0.0006776
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 6, lower bound: -0.0006791, upper bound: 0.0006776
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 6, lower bound: -0.0006791, upper bound: 0.0006776
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 6, lower bound: -0.0006791, upper bound: 0.0006776
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 6, lower bound: -0.0006791, upper bound: 0.0006776
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 6, lower bound: -0.0006791, upper bound: 0.0006776

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.50 + 47.94 = 51.44 seconds
