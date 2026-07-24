## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00640413


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764)
1: (-0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001)
2: (0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567)
3: (0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0122357, 0.0122357)
4: (-0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686)
5: (0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359)
6: (0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770)
7: (-0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623)
8: (0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190)
9: (0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.12 + 1.89 = 3.01 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0071157, upper bound: 0.0071157

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065253, upper bound: 0.0065253
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065253, upper bound: 0.0065253
time: 0.86 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.73 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.73
Output dim: 2, lower bound: -0.0065253, upper bound: 0.0065253
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.73
Output dim: 2, lower bound: -0.0065253, upper bound: 0.0065253

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0120643, 0.0121869
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0063117, upper bound: 0.0063117
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0063117, upper bound: 0.0063117
time: 1.02 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0122357, 0.0120643
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0063117, upper bound: 0.0063117
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0063117, upper bound: 0.0063117
time: 1.03 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.00 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.00
Output dim: 2, lower bound: -0.0063117, upper bound: 0.0063117
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.00
Output dim: 2, lower bound: -0.0063117, upper bound: 0.0063117
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.00
Output dim: 2, lower bound: -0.0063117, upper bound: 0.0063117
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.00
Output dim: 2, lower bound: -0.0063117, upper bound: 0.0063117

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.01 + 9.71 = 12.72 seconds
