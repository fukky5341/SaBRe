## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00045437


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823)
1: (0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037)
2: (0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0045441, 0.0045441)
3: (-0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783)
4: (0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0007265, 0.0007265)
5: (-0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838)
6: (-0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170)
7: (-0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583)
8: (-0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360)
9: (1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.27 + 1.49 = 2.76 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0005328, upper bound: 0.0005328

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005130, upper bound: 0.0005130
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005130, upper bound: 0.0005130
time: 0.66 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.36 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.36
Output dim: 9, lower bound: -0.0005130, upper bound: 0.0005130
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.36
Output dim: 9, lower bound: -0.0005130, upper bound: 0.0005130

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0045383, 0.0045407
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0007236, 0.0007220
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005078, upper bound: 0.0005078
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005078, upper bound: 0.0005078
time: 1.01 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0045441, 0.0045383
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0007220, 0.0007265
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005085, upper bound: 0.0005085
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005085, upper bound: 0.0005085
time: 0.79 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.61 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.61
Output dim: 9, lower bound: -0.0005078, upper bound: 0.0005078
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.61
Output dim: 9, lower bound: -0.0005078, upper bound: 0.0005078
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.61
Output dim: 9, lower bound: -0.0005085, upper bound: 0.0005085
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.61
Output dim: 9, lower bound: -0.0005085, upper bound: 0.0005085

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0045110, 0.0045115
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0007069, 0.0007122
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005074, upper bound: 0.0005069
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005069, upper bound: 0.0005074
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0045091, 0.0045407
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0007236, 0.0007053
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005031, upper bound: 0.0005031
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005031, upper bound: 0.0005031
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0044263, 0.0044377
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0007002, 0.0006921
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005031, upper bound: 0.0005031
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005031, upper bound: 0.0005031
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0044435, 0.0044206
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006873, 0.0007050
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005081, upper bound: 0.0005077
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005077, upper bound: 0.0005081
time: 0.86 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.58 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.58
Output dim: 9, lower bound: -0.0005074, upper bound: 0.0005069
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.58
Output dim: 9, lower bound: -0.0005069, upper bound: 0.0005074
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.58
Output dim: 9, lower bound: -0.0005031, upper bound: 0.0005031
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.58
Output dim: 9, lower bound: -0.0005031, upper bound: 0.0005031
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.58
Output dim: 9, lower bound: -0.0005031, upper bound: 0.0005031
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.58
Output dim: 9, lower bound: -0.0005031, upper bound: 0.0005031
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.58
Output dim: 9, lower bound: -0.0005081, upper bound: 0.0005077
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.58
Output dim: 9, lower bound: -0.0005077, upper bound: 0.0005081

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043187, 0.0043861
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006789, 0.0006581
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005072, upper bound: 0.0005067
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005072, upper bound: 0.0005067
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043836, 0.0043191
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006528, 0.0006842
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005056, upper bound: 0.0005060
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005056, upper bound: 0.0005060
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043916, 0.0044404
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0007020, 0.0006699
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005016, upper bound: 0.0005016
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005016, upper bound: 0.0005016
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0044088, 0.0044230
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006889, 0.0006827
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005029, upper bound: 0.0005029
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005029, upper bound: 0.0005029
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043993, 0.0044088
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006827, 0.0006816
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005029, upper bound: 0.0005029
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005029, upper bound: 0.0005029
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043974, 0.0044377
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0007002, 0.0006747
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005027, upper bound: 0.0005025
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005025, upper bound: 0.0005027
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042509, 0.0042942
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006511, 0.0006428
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005027, upper bound: 0.0005025
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005027, upper bound: 0.0005025
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043185, 0.0042281
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006252, 0.0006683
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005064, upper bound: 0.0005068
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005064, upper bound: 0.0005068
time: 0.79 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.67 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 9, lower bound: -0.0005072, upper bound: 0.0005067
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 9, lower bound: -0.0005072, upper bound: 0.0005067
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 9, lower bound: -0.0005056, upper bound: 0.0005060
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 9, lower bound: -0.0005056, upper bound: 0.0005060
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 9, lower bound: -0.0005016, upper bound: 0.0005016
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 9, lower bound: -0.0005016, upper bound: 0.0005016
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 9, lower bound: -0.0005029, upper bound: 0.0005029
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 9, lower bound: -0.0005029, upper bound: 0.0005029
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 9, lower bound: -0.0005029, upper bound: 0.0005029
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 9, lower bound: -0.0005029, upper bound: 0.0005029
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 9, lower bound: -0.0005027, upper bound: 0.0005025
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 9, lower bound: -0.0005025, upper bound: 0.0005027
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 9, lower bound: -0.0005027, upper bound: 0.0005025
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 9, lower bound: -0.0005027, upper bound: 0.0005025
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 9, lower bound: -0.0005064, upper bound: 0.0005068
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 9, lower bound: -0.0005064, upper bound: 0.0005068

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042557, 0.0043103
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006393, 0.0006320
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004931, upper bound: 0.0004927
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004931, upper bound: 0.0004927
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042429, 0.0043861
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006789, 0.0006184
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005025, upper bound: 0.0005022
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005025, upper bound: 0.0005022
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043758, 0.0043135
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006474, 0.0006783
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005053, upper bound: 0.0005057
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005053, upper bound: 0.0005057
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043774, 0.0043113
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006468, 0.0006794
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004913, upper bound: 0.0004917
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004913, upper bound: 0.0004917
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043844, 0.0044347
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006953, 0.0006618
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005014, upper bound: 0.0005014
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005014, upper bound: 0.0005014
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043862, 0.0044331
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006942, 0.0006627
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005014, upper bound: 0.0005014
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005014, upper bound: 0.0005014
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043292, 0.0043449
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006484, 0.0006461
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004882, upper bound: 0.0004882
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004882, upper bound: 0.0004882
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043309, 0.0044230
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006889, 0.0006420
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004882, upper bound: 0.0004882
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004882, upper bound: 0.0004882
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043343, 0.0043309
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006420, 0.0006544
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004882, upper bound: 0.0004882
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004882, upper bound: 0.0004882
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043215, 0.0044088
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006827, 0.0006408
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005014, upper bound: 0.0005014
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005014, upper bound: 0.0005014
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042060, 0.0043128
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006635, 0.0006129
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005025, upper bound: 0.0005022
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005025, upper bound: 0.0005022
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042722, 0.0042452
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006380, 0.0006389
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005022, upper bound: 0.0005025
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005022, upper bound: 0.0005025
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042254, 0.0042665
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006341, 0.0006300
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005013, upper bound: 0.0005010
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005013, upper bound: 0.0005010
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042232, 0.0042942
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006511, 0.0006258
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005013, upper bound: 0.0005010
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005013, upper bound: 0.0005010
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043108, 0.0042222
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006204, 0.0006626
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005011, upper bound: 0.0005013
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005010, upper bound: 0.0005013
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043126, 0.0042203
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006194, 0.0006634
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005011, upper bound: 0.0005013
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005011, upper bound: 0.0005013
time: 0.84 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.67 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 9, lower bound: -0.0004931, upper bound: 0.0004927
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 9, lower bound: -0.0004931, upper bound: 0.0004927
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 9, lower bound: -0.0005025, upper bound: 0.0005022
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 9, lower bound: -0.0005025, upper bound: 0.0005022
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 9, lower bound: -0.0005053, upper bound: 0.0005057
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 9, lower bound: -0.0005053, upper bound: 0.0005057
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 9, lower bound: -0.0004913, upper bound: 0.0004917
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 9, lower bound: -0.0004913, upper bound: 0.0004917
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 9, lower bound: -0.0005014, upper bound: 0.0005014
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 9, lower bound: -0.0005014, upper bound: 0.0005014
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 9, lower bound: -0.0005014, upper bound: 0.0005014
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 9, lower bound: -0.0005014, upper bound: 0.0005014
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 9, lower bound: -0.0004882, upper bound: 0.0004882
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 9, lower bound: -0.0004882, upper bound: 0.0004882
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 9, lower bound: -0.0004882, upper bound: 0.0004882
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 9, lower bound: -0.0004882, upper bound: 0.0004882
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 9, lower bound: -0.0004882, upper bound: 0.0004882
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 9, lower bound: -0.0004882, upper bound: 0.0004882
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 9, lower bound: -0.0005014, upper bound: 0.0005014
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 9, lower bound: -0.0005014, upper bound: 0.0005014
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 9, lower bound: -0.0005025, upper bound: 0.0005022
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 9, lower bound: -0.0005025, upper bound: 0.0005022
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 9, lower bound: -0.0005022, upper bound: 0.0005025
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 9, lower bound: -0.0005022, upper bound: 0.0005025
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 9, lower bound: -0.0005013, upper bound: 0.0005010
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 9, lower bound: -0.0005013, upper bound: 0.0005010
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 9, lower bound: -0.0005013, upper bound: 0.0005010
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 9, lower bound: -0.0005013, upper bound: 0.0005010
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 9, lower bound: -0.0005011, upper bound: 0.0005013
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 9, lower bound: -0.0005010, upper bound: 0.0005013
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 9, lower bound: -0.0005011, upper bound: 0.0005013
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 9, lower bound: -0.0005011, upper bound: 0.0005013

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042393, 0.0042993
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022711, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006366, 0.0006306
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004878, upper bound: 0.0004876
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004878, upper bound: 0.0004876
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042557, 0.0042939
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006393, 0.0006293
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004878, upper bound: 0.0004876
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004878, upper bound: 0.0004876
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041231, 0.0042879
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022763
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006487, 0.0005725
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004878, upper bound: 0.0004876
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004878, upper bound: 0.0004876
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041405, 0.0042697
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022782
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006359, 0.0005826
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005011, upper bound: 0.0005008
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005011, upper bound: 0.0005008
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043173, 0.0042379
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022747, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006084, 0.0006519
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004913, upper bound: 0.0004915
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004913, upper bound: 0.0004915
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043002, 0.0043135
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006474, 0.0006393
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005008, upper bound: 0.0005011
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005008, upper bound: 0.0005010
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043610, 0.0042993
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006440, 0.0006773
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004913, upper bound: 0.0004916
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004913, upper bound: 0.0004916
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043774, 0.0042949
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006468, 0.0006767
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004863
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004862
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043121, 0.0043564
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006554, 0.0006327
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005011, upper bound: 0.0005008
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005008, upper bound: 0.0005011
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043062, 0.0044347
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022739
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006953, 0.0006223
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004867, upper bound: 0.0004866
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004867, upper bound: 0.0004866
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043140, 0.0043548
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006542, 0.0006345
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004866, upper bound: 0.0004867
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004866, upper bound: 0.0004867
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043080, 0.0044331
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022745
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006942, 0.0006232
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005010, upper bound: 0.0005008
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005007, upper bound: 0.0005011
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043126, 0.0043327
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006456, 0.0006445
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004878, upper bound: 0.0004876
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004876, upper bound: 0.0004878
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043292, 0.0043283
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006484, 0.0006435
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004867, upper bound: 0.0004866
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004866, upper bound: 0.0004867
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043144, 0.0044108
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006864, 0.0006395
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004867, upper bound: 0.0004866
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004866, upper bound: 0.0004867
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043309, 0.0044064
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022730
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006889, 0.0006394
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004878, upper bound: 0.0004876
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004876, upper bound: 0.0004878
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043178, 0.0043185
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022730, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006394, 0.0006530
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004867, upper bound: 0.0004866
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004866, upper bound: 0.0004867
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043343, 0.0043144
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006420, 0.0006518
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004867, upper bound: 0.0004866
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004866, upper bound: 0.0004867
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043139, 0.0044032
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006759, 0.0006340
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004867, upper bound: 0.0004866
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004867, upper bound: 0.0004866
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043153, 0.0044015
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006746, 0.0006350
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005010, upper bound: 0.0005008
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005008, upper bound: 0.0005011
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041328, 0.0042342
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006205, 0.0005808
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005011, upper bound: 0.0005008
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005010, upper bound: 0.0005008
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041268, 0.0043128
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022736
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006635, 0.0005704
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004878, upper bound: 0.0004876
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004878, upper bound: 0.0004876
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041981, 0.0041666
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0005950, 0.0006055
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004876, upper bound: 0.0004878
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004876, upper bound: 0.0004878
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041930, 0.0042452
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022738
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006380, 0.0005963
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005008, upper bound: 0.0005011
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005007, upper bound: 0.0005011
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042176, 0.0042608
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006293, 0.0006242
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004861
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004861
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042187, 0.0042587
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006283, 0.0006253
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004862
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004862
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042154, 0.0042886
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006464, 0.0006200
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004861
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004861
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042171, 0.0042865
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006454, 0.0006212
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004862
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004862
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042854, 0.0041944
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006033, 0.0006506
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004862
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004862
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042830, 0.0042222
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006204, 0.0006455
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005008, upper bound: 0.0005010
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005008, upper bound: 0.0005010
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042868, 0.0041926
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006024, 0.0006517
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005008, upper bound: 0.0005011
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005008, upper bound: 0.0005011
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042848, 0.0042203
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006194, 0.0006464
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005008, upper bound: 0.0005011
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005008, upper bound: 0.0005011
time: 0.84 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.94 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004878, upper bound: 0.0004876
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004878, upper bound: 0.0004876
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004878, upper bound: 0.0004876
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004878, upper bound: 0.0004876
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004878, upper bound: 0.0004876
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004878, upper bound: 0.0004876
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0005011, upper bound: 0.0005008
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0005011, upper bound: 0.0005008
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004913, upper bound: 0.0004915
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004913, upper bound: 0.0004915
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0005008, upper bound: 0.0005011
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0005008, upper bound: 0.0005010
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004913, upper bound: 0.0004916
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004913, upper bound: 0.0004916
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004863
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004862
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0005011, upper bound: 0.0005008
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0005008, upper bound: 0.0005011
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004867, upper bound: 0.0004866
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004867, upper bound: 0.0004866
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004866, upper bound: 0.0004867
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004866, upper bound: 0.0004867
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0005010, upper bound: 0.0005008
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0005007, upper bound: 0.0005011
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004878, upper bound: 0.0004876
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004876, upper bound: 0.0004878
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004867, upper bound: 0.0004866
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004866, upper bound: 0.0004867
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004867, upper bound: 0.0004866
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004866, upper bound: 0.0004867
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004878, upper bound: 0.0004876
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004876, upper bound: 0.0004878
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004867, upper bound: 0.0004866
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004866, upper bound: 0.0004867
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004867, upper bound: 0.0004866
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004866, upper bound: 0.0004867
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004867, upper bound: 0.0004866
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004867, upper bound: 0.0004866
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0005010, upper bound: 0.0005008
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0005008, upper bound: 0.0005011
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0005011, upper bound: 0.0005008
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0005010, upper bound: 0.0005008
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004878, upper bound: 0.0004876
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004878, upper bound: 0.0004876
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004876, upper bound: 0.0004878
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004876, upper bound: 0.0004878
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0005008, upper bound: 0.0005011
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0005007, upper bound: 0.0005011
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004861
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004861
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004862
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004862
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004861
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004861
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004862
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004862
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004862
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004862
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0005008, upper bound: 0.0005010
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0005008, upper bound: 0.0005010
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0005008, upper bound: 0.0005011
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0005008, upper bound: 0.0005011
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0005008, upper bound: 0.0005011
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 9, lower bound: -0.0005008, upper bound: 0.0005011

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041192, 0.0041957
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022636, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006036, 0.0005847
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004860
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041266, 0.0041791
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022615, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0005908, 0.0005903
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004860
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004861
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041359, 0.0041920
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022730, 0.0022745
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006061, 0.0005835
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004860
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041433, 0.0041738
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022710, 0.0022759
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0005933, 0.0005888
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004860
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004861
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041063, 0.0042749
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022750
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006462, 0.0005709
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004860
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041231, 0.0042712
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022669
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006487, 0.0005699
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004860
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041328, 0.0042640
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022719
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006310, 0.0005787
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004860
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004860
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041339, 0.0042619
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022729
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006301, 0.0005798
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004861
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004861
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043009, 0.0042259
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022653, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006058, 0.0006502
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004862
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043173, 0.0042215
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022747, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006084, 0.0006492
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004862
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041803, 0.0042140
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022704
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006181, 0.0005947
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004862
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004862
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042006, 0.0041971
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022722
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006046, 0.0006051
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043027, 0.0042238
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022648, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006052, 0.0006516
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004863
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004862
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042854, 0.0042993
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006440, 0.0006385
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004863
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004862
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042610, 0.0041957
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006170, 0.0006339
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004862
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004863
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042811, 0.0041783
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006040, 0.0006443
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004862
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004862
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041194, 0.0042310
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022759, 0.0022710
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006194, 0.0005721
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004860
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004860
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041846, 0.0041630
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022756, 0.0022719
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0005939, 0.0005968
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004862
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004862
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042897, 0.0044234
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022724
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006925, 0.0006203
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004860
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004862
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043062, 0.0044182
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022645
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006953, 0.0006196
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004860
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004862
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042975, 0.0043433
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022755, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006515, 0.0006328
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004863
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043140, 0.0043383
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022721
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006542, 0.0006318
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004863
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041153, 0.0043079
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022644
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006600, 0.0005626
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041816, 0.0042402
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022648
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006341, 0.0005886
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004863
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004863
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041198, 0.0042075
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022698, 0.0022778
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006074, 0.0005810
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004860
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004861
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041888, 0.0041395
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022697, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0005813, 0.0006040
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004862
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043216, 0.0043230
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022731
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006418, 0.0006365
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004860
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043237, 0.0043209
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022738
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006412, 0.0006382
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004861
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004862
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043068, 0.0044057
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022749
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006790, 0.0006326
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004860
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043085, 0.0044036
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022756
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006784, 0.0006336
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004861
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004862
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041382, 0.0042807
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022631
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006529, 0.0005759
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004860
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004861
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042058, 0.0042137
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022632
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006268, 0.0006014
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004862
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043102, 0.0043128
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022674, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006337, 0.0006461
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004860
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004862
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043120, 0.0043109
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022668, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006324, 0.0006478
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004863
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043267, 0.0043085
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022768, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006363, 0.0006449
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004860
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004862
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043285, 0.0043068
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022762, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006351, 0.0006465
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004863
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042973, 0.0043910
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006732, 0.0006323
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004860
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004862
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0043139, 0.0043867
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022739
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006759, 0.0006313
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004860
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004862
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041224, 0.0042773
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022742
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006407, 0.0005743
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041876, 0.0042097
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022745
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006152, 0.0006007
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004863
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004863
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041251, 0.0042282
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022755, 0.0022744
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006173, 0.0005769
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004860
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004860
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041269, 0.0042263
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022748, 0.0022749
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006165, 0.0005787
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041101, 0.0042997
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022721
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006608, 0.0005686
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004860
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041268, 0.0042960
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022642
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006635, 0.0005678
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004860
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041813, 0.0041540
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022714, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0005925, 0.0006037
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004862
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004863
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041981, 0.0041499
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022722
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0005950, 0.0006029
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004862
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004863
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041853, 0.0042392
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022675
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006335, 0.0005924
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004862
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004862
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041874, 0.0042375
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022682
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006323, 0.0005934
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004863
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004863
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042010, 0.0042499
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022721, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006267, 0.0006223
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004860
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004860
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042176, 0.0042441
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006293, 0.0006215
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004860
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004860
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042021, 0.0042478
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022713, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006257, 0.0006235
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004861
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004861
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042187, 0.0042421
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006283, 0.0006227
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004861
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004861
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041987, 0.0042777
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006436, 0.0006175
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004860
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004860
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042154, 0.0042719
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022768
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006464, 0.0006173
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004860
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004860
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042004, 0.0042756
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006426, 0.0006185
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004861
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004861
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042171, 0.0042698
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022774
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006454, 0.0006186
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004861
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004861
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042688, 0.0041818
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022717, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006007, 0.0006480
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042854, 0.0041778
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006033, 0.0006479
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042036, 0.0041435
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022729, 0.0022766
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0005791, 0.0006065
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042038, 0.0042222
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022697
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006204, 0.0006048
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042138, 0.0041134
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022639, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0005617, 0.0006182
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004862
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004862
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042076, 0.0041926
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022767
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006024, 0.0006110
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004862
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004862
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042058, 0.0041417
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022724, 0.0022773
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0005781, 0.0006083
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004862
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004862
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042057, 0.0042203
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022704
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006194, 0.0006057
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004862
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004862
time: 0.89 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.00 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004860
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004860
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004861
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004860
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004860
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004861
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004860
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004860
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004860
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004860
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004861
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004861
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004862
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004862
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004862
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004862
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004863
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004862
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004863
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004862
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004862
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004863
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004862
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004862
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004860
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004860
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004862
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004862
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004860
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004862
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004860
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004862
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004863
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004863
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004863
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004863
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004860
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004861
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004862
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004860
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004861
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004862
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004860
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004861
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004862
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004860
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004861
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004862
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004860
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004862
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004863
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004860
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004862
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004863
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004860
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004862
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004860
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004862
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004863
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004863
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004860
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004860
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004860
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004860
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004862
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004863
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004862
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004863
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004862
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004862
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004863
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004863
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004860
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004860
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004860
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004860
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004861
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004861
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004861
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004861
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004860
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004860
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004860
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004860
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004861
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004861
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004861
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004861
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004861
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004862
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004862
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004862
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004862
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004862
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004862
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004862
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004862

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041115, 0.0041899
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022580, 0.0022765
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006003, 0.0005807
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 124
type: RSZ, layer: 3, pos: 121

Time for candidate selection: 1.89 seconds

### Candidate
type: RSZ, layer: 3, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004851, upper bound: 0.0004846
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004851, upper bound: 0.0004846
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041133, 0.0041881
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022572, 0.0022775
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0005996, 0.0005824
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 124

Time for candidate selection: 1.91 seconds

### Candidate
type: RSZ, layer: 3, pos: 121

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004855, upper bound: 0.0004861
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004854
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041190, 0.0041736
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022558, 0.0022781
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0005877, 0.0005863
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 124
type: RSZ, layer: 3, pos: 121

Time for candidate selection: 1.93 seconds

### Candidate
type: RSZ, layer: 3, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004851, upper bound: 0.0004846
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004851, upper bound: 0.0004846
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041205, 0.0041715
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022552, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0005868, 0.0005878
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 124
type: RSZ, layer: 3, pos: 121

Time for candidate selection: 1.88 seconds

### Candidate
type: RSZ, layer: 3, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004850, upper bound: 0.0004847
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004850, upper bound: 0.0004847
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041282, 0.0041861
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022674, 0.0022681
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006029, 0.0005795
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 124
type: RSZ, layer: 3, pos: 121

Time for candidate selection: 1.88 seconds

### Candidate
type: RSZ, layer: 3, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004851, upper bound: 0.0004846
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004851, upper bound: 0.0004846
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041300, 0.0041843
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022667, 0.0022690
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006022, 0.0005811
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 124

Time for candidate selection: 1.91 seconds

### Candidate
type: RSZ, layer: 3, pos: 121

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004855, upper bound: 0.0004861
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004854
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041356, 0.0041682
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022653, 0.0022696
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0005903, 0.0005848
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 124
type: RSZ, layer: 3, pos: 121

Time for candidate selection: 1.91 seconds

### Candidate
type: RSZ, layer: 3, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004851, upper bound: 0.0004846
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004851, upper bound: 0.0004846
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041371, 0.0041661
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022646, 0.0022705
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0005894, 0.0005863
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 124

Time for candidate selection: 1.90 seconds

### Candidate
type: RSZ, layer: 3, pos: 121

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004855, upper bound: 0.0004861
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004854
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0040987, 0.0042691
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022747, 0.0022686
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006410, 0.0005669
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 124
type: RSZ, layer: 3, pos: 121

Time for candidate selection: 1.92 seconds

### Candidate
type: RSZ, layer: 3, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004851, upper bound: 0.0004846
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004851, upper bound: 0.0004846
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041001, 0.0042673
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022739, 0.0022696
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006402, 0.0005679
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 124

Time for candidate selection: 1.92 seconds

### Candidate
type: RSZ, layer: 3, pos: 121

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004855, upper bound: 0.0004861
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004854
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041153, 0.0042652
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022605
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006436, 0.0005659
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 124
type: RSZ, layer: 3, pos: 121

Time for candidate selection: 1.90 seconds

### Candidate
type: RSZ, layer: 3, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004851, upper bound: 0.0004846
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004851, upper bound: 0.0004846
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041167, 0.0042635
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022614
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006429, 0.0005669
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 124

Time for candidate selection: 1.94 seconds

### Candidate
type: RSZ, layer: 3, pos: 121

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004855, upper bound: 0.0004861
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004854
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041161, 0.0042528
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022725, 0.0022709
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006283, 0.0005768
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 124

Time for candidate selection: 1.93 seconds

### Candidate
type: RSZ, layer: 3, pos: 121

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004855, upper bound: 0.0004860
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004863, upper bound: 0.0004854
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041328, 0.0042474
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022624
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006310, 0.0005761
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 124
type: RSZ, layer: 3, pos: 121

Time for candidate selection: 1.88 seconds

### Candidate
type: RSZ, layer: 3, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004851, upper bound: 0.0004846
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004851, upper bound: 0.0004846
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041172, 0.0042507
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022719, 0.0022721
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006274, 0.0005780
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 124
type: RSZ, layer: 3, pos: 121

Time for candidate selection: 1.87 seconds

### Candidate
type: RSZ, layer: 3, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004850, upper bound: 0.0004847
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004850, upper bound: 0.0004847
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041339, 0.0042453
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022635
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006301, 0.0005772
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 124

Time for candidate selection: 1.88 seconds

### Candidate
type: RSZ, layer: 3, pos: 121

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004855, upper bound: 0.0004861
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004854
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041807, 0.0041234
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022577, 0.0022777
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0005748, 0.0006056
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 124
type: RSZ, layer: 3, pos: 121

Time for candidate selection: 1.89 seconds

### Candidate
type: RSZ, layer: 3, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004847, upper bound: 0.0004850
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004847, upper bound: 0.0004850
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041894, 0.0041056
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022556, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0005612, 0.0006103
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 124

Time for candidate selection: 1.84 seconds

### Candidate
type: RSZ, layer: 3, pos: 121

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004854, upper bound: 0.0004861
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004855
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041973, 0.0041181
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022671, 0.0022688
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0005774, 0.0006046
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 124

Time for candidate selection: 1.89 seconds

### Candidate
type: RSZ, layer: 3, pos: 121

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004854, upper bound: 0.0004862
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004855
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042061, 0.0041013
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022650, 0.0022703
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0005639, 0.0006092
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 124

Time for candidate selection: 1.89 seconds

### Candidate
type: RSZ, layer: 3, pos: 121

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004854, upper bound: 0.0004861
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004855
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041636, 0.0042025
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022743, 0.0022695
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006155, 0.0005927
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 124
type: RSZ, layer: 3, pos: 121

Time for candidate selection: 1.88 seconds

### Candidate
type: RSZ, layer: 3, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004847, upper bound: 0.0004850
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004847, upper bound: 0.0004850
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041803, 0.0041973
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022610
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006181, 0.0005921
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 124

Time for candidate selection: 1.87 seconds

### Candidate
type: RSZ, layer: 3, pos: 121

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004854, upper bound: 0.0004862
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004855
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041839, 0.0041848
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022722, 0.0022720
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006019, 0.0006026
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 124

Time for candidate selection: 1.88 seconds

### Candidate
type: RSZ, layer: 3, pos: 121

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004854, upper bound: 0.0004861
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004855
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042006, 0.0041804
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022628
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006046, 0.0006025
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 124

Time for candidate selection: 1.90 seconds

### Candidate
type: RSZ, layer: 3, pos: 121

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004854, upper bound: 0.0004861
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004861, upper bound: 0.0004855
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041824, 0.0041216
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022571, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0005737, 0.0006070
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 124

Time for candidate selection: 1.88 seconds

### Candidate
type: RSZ, layer: 3, pos: 121

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004854, upper bound: 0.0004863
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004855
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041914, 0.0041035
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022551, 0.0022783
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0005606, 0.0006117
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 124

Time for candidate selection: 1.89 seconds

### Candidate
type: RSZ, layer: 3, pos: 121

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004854, upper bound: 0.0004862
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004855
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041652, 0.0042008
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022738, 0.0022706
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006143, 0.0005940
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 124

Time for candidate selection: 1.95 seconds

### Candidate
type: RSZ, layer: 3, pos: 121

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004854, upper bound: 0.0004863
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004860, upper bound: 0.0004855
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041853, 0.0041827
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022717, 0.0022732
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006013, 0.0006038
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 124
type: RSZ, layer: 3, pos: 121

Time for candidate selection: 1.95 seconds

### Candidate
type: RSZ, layer: 3, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004846, upper bound: 0.0004851
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004846, upper bound: 0.0004851
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041991, 0.0041166
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022665, 0.0022697
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0005763, 0.0006060
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 124
type: RSZ, layer: 3, pos: 121

Time for candidate selection: 1.90 seconds

### Candidate
type: RSZ, layer: 3, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004846, upper bound: 0.0004851
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004846, upper bound: 0.0004851
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0041819, 0.0041957
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022783, 0.0022617
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0006170, 0.0005933
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 124
type: RSZ, layer: 3, pos: 121

Time for candidate selection: 1.87 seconds

### Candidate
type: RSZ, layer: 3, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004846, upper bound: 0.0004851
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004846, upper bound: 0.0004851
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041196, -0.0014373, -0.0041196, -0.0014373, -0.0026823, 0.0026823
1: 0.0047631, 0.0066668, 0.0047631, 0.0066668, -0.0019037, 0.0019037
2: 0.0103725, 0.0154128, 0.0103725, 0.0154128, -0.0042081, 0.0040991
3: -0.0049861, -0.0027078, -0.0049861, -0.0027078, -0.0022645, 0.0022713
4: 0.0044675, 0.0052266, 0.0044675, 0.0052266, -0.0005633, 0.0006108
5: -0.0025335, -0.0008497, -0.0025335, -0.0008497, -0.0016838, 0.0016838
6: -0.0061155, -0.0052984, -0.0061155, -0.0052984, -0.0008170, 0.0008170
7: -0.0032928, -0.0017344, -0.0032928, -0.0017344, -0.0015583, 0.0015583
8: -0.0046717, -0.0013357, -0.0046717, -0.0013357, -0.0033360, 0.0033360
9: 1.0004220, 1.0009845, 1.0004220, 1.0009845, -0.0005625, 0.0005625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 2.76 + 597.29 = 600.05 seconds
