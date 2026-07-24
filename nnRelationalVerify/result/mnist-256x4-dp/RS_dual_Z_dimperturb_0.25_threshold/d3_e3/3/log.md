## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.000229875


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0130157, -0.0111796, -0.0130157, -0.0111796, -0.0011395, 0.0011395)
1: (-0.0066083, -0.0060906, -0.0066083, -0.0060906, -0.0003213, 0.0003213)
2: (-0.0101975, -0.0063780, -0.0101975, -0.0063780, -0.0023705, 0.0023705)
3: (0.0002778, 0.0007833, 0.0002778, 0.0007833, -0.0003137, 0.0003137)
4: (0.0108584, 0.0137128, 0.0108584, 0.0137128, -0.0017716, 0.0017716)
5: (0.9985231, 0.9993161, 0.9985231, 0.9993161, -0.0004922, 0.0004922)
6: (0.0065430, 0.0072628, 0.0065430, 0.0072628, -0.0004468, 0.0004468)
7: (0.0010358, 0.0037222, 0.0010358, 0.0037222, -0.0016672, 0.0016672)
8: (-0.0120898, -0.0099990, -0.0120898, -0.0099990, -0.0012976, 0.0012976)
9: (-0.0031471, -0.0029667, -0.0031471, -0.0029667, -0.0001120, 0.0001120)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.66 + 1.39 = 3.05 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0003108, upper bound: 0.0003109

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003060, upper bound: 0.0003041
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003040, upper bound: 0.0003061
time: 0.63 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.37 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.37
Output dim: 5, lower bound: -0.0003060, upper bound: 0.0003041
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.37
Output dim: 5, lower bound: -0.0003040, upper bound: 0.0003061

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0130157, -0.0111796, -0.0130157, -0.0111796, -0.0011064, 0.0011068
1: -0.0066083, -0.0060906, -0.0066083, -0.0060906, -0.0003119, 0.0003121
2: -0.0101975, -0.0063780, -0.0101975, -0.0063780, -0.0023015, 0.0023024
3: 0.0002778, 0.0007833, 0.0002778, 0.0007833, -0.0003046, 0.0003047
4: 0.0108584, 0.0137128, 0.0108584, 0.0137128, -0.0017207, 0.0017200
5: 0.9985231, 0.9993161, 0.9985231, 0.9993161, -0.0004781, 0.0004779
6: 0.0065430, 0.0072628, 0.0065430, 0.0072628, -0.0004339, 0.0004338
7: 0.0010358, 0.0037222, 0.0010358, 0.0037222, -0.0016194, 0.0016187
8: -0.0120898, -0.0099990, -0.0120898, -0.0099990, -0.0012599, 0.0012604
9: -0.0031471, -0.0029667, -0.0031471, -0.0029667, -0.0001087, 0.0001087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001392, upper bound: 0.0001384
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001392, upper bound: 0.0001384
time: 0.51 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0130157, -0.0111796, -0.0130157, -0.0111796, -0.0011068, 0.0011064
1: -0.0066083, -0.0060906, -0.0066083, -0.0060906, -0.0003121, 0.0003119
2: -0.0101975, -0.0063780, -0.0101975, -0.0063780, -0.0023024, 0.0023015
3: 0.0002778, 0.0007833, 0.0002778, 0.0007833, -0.0003047, 0.0003046
4: 0.0108584, 0.0137128, 0.0108584, 0.0137128, -0.0017200, 0.0017207
5: 0.9985231, 0.9993161, 0.9985231, 0.9993161, -0.0004779, 0.0004781
6: 0.0065430, 0.0072628, 0.0065430, 0.0072628, -0.0004338, 0.0004339
7: 0.0010358, 0.0037222, 0.0010358, 0.0037222, -0.0016187, 0.0016194
8: -0.0120898, -0.0099990, -0.0120898, -0.0099990, -0.0012604, 0.0012599
9: -0.0031471, -0.0029667, -0.0031471, -0.0029667, -0.0001087, 0.0001087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001384, upper bound: 0.0001392
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001384, upper bound: 0.0001392
time: 0.52 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.65 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.65
Output dim: 5, lower bound: -0.0001392, upper bound: 0.0001384
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.65
Output dim: 5, lower bound: -0.0001392, upper bound: 0.0001384
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.65
Output dim: 5, lower bound: -0.0001384, upper bound: 0.0001392
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.65
Output dim: 5, lower bound: -0.0001384, upper bound: 0.0001392

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.05 + 6.75 = 9.80 seconds
