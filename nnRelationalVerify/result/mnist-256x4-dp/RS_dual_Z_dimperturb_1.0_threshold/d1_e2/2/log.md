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
execution time: IAR + RelationalAnalysis = 1.14 + 1.90 = 3.04 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0071157, upper bound: 0.0071157

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0069084
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0069084
time: 1.29 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.50 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.50
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0069084
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.50
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0069084

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0121205, 0.0119585
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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067783, upper bound: 0.0067783
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067783, upper bound: 0.0067783
time: 1.12 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0119585, 0.0122357
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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067783, upper bound: 0.0067783
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067783, upper bound: 0.0067783
time: 1.08 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.33 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.33
Output dim: 2, lower bound: -0.0067783, upper bound: 0.0067783
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.33
Output dim: 2, lower bound: -0.0067783, upper bound: 0.0067783
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.33
Output dim: 2, lower bound: -0.0067783, upper bound: 0.0067783
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.33
Output dim: 2, lower bound: -0.0067783, upper bound: 0.0067783

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0120895, 0.0119192
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067783, upper bound: 0.0067783
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067783, upper bound: 0.0067783
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0120812, 0.0119585
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067783, upper bound: 0.0067783
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067783, upper bound: 0.0067783
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0119296, 0.0121946
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067783, upper bound: 0.0067783
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067783, upper bound: 0.0067783
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0119192, 0.0122357
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067783, upper bound: 0.0067783
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067783, upper bound: 0.0067783
time: 1.16 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.56 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 2, lower bound: -0.0067783, upper bound: 0.0067783
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 2, lower bound: -0.0067783, upper bound: 0.0067783
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 2, lower bound: -0.0067783, upper bound: 0.0067783
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 2, lower bound: -0.0067783, upper bound: 0.0067783
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 2, lower bound: -0.0067783, upper bound: 0.0067783
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 2, lower bound: -0.0067783, upper bound: 0.0067783
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 2, lower bound: -0.0067783, upper bound: 0.0067783
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 2, lower bound: -0.0067783, upper bound: 0.0067783

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0106599, 0.0108738
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0110167, 0.0104895
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0106516, 0.0109085
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0110132, 0.0105242
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0104999, 0.0111833
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0108806, 0.0107990
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0104895, 0.0112183
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0108738, 0.0108340
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.24 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.63 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0104115, 0.0107672
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0106599, 0.0106255
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0107683, 0.0103328
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0066742, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0110167, 0.0102412
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0066127, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0104032, 0.0108018
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067215

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0106516, 0.0106602
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0107648, 0.0103674
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0066982, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0110132, 0.0102759
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0066367, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0102516, 0.0110766
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0066197

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0104999, 0.0109350
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0106323, 0.0106422
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0108806, 0.0105506
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0102412, 0.0111117
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0066127

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0104895, 0.0109700
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0106255, 0.0106773
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0108738, 0.0105857
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.31 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.73 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0101611, 0.0106052
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0066617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0102247, 0.0105167
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067043

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0104021, 0.0104566
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0104656, 0.0103751
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0105179, 0.0101567
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0066587, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0105977, 0.0100823
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0066088, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0107589, 0.0100568
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0065916, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0108387, 0.0099907
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0065473, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0101528, 0.0106412
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0066561

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0102167, 0.0105527
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0066990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0103938, 0.0104926
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0104577, 0.0104111
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0105144, 0.0101927
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0066830, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0105935, 0.0101183
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0066331, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0107554, 0.0100928
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0066159, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0108345, 0.0100268
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0065716, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0100011, 0.0109166
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0065543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0100637, 0.0108281
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0065963

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0102421, 0.0107680
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0066966

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0103047, 0.0106864
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0103819, 0.0104681
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0104606, 0.0103937
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0106228, 0.0103681
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0107016, 0.0103021
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0099907, 0.0109535
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0065473

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0100568, 0.0108650
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0065916

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0102317, 0.0108048
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0066896

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0102977, 0.0107233
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0103751, 0.0105049
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0104566, 0.0104306
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0106161, 0.0104050
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0106976, 0.0103390
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
time: 1.22 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.69 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 2, lower bound: -0.0066823, upper bound: 0.0066823

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0098216, 0.0103391
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0065525

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0098665, 0.0102657
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0065827

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0098851, 0.0102550
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0065952

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0099278, 0.0101772
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0066238

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0100563, 0.0101793
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0066910

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0101012, 0.0101171
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067212

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0101198, 0.0100983
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0101625, 0.0100355
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0066962, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0101784, 0.0098773
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0065899, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0102377, 0.0098172
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0065495, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0102582, 0.0098075
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0065431, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0103151, 0.0097428
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0064996, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0104131, 0.0097653
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0065147, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0104724, 0.0097172
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0064825, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0104929, 0.0097016
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0064719, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0105498, 0.0096512
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0064381, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0098133, 0.0103754
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0065469

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0098588, 0.0103021
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0065775

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0098772, 0.0102914
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0065898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0099211, 0.0102135
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0066193

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0100480, 0.0102156
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0066854

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0100935, 0.0101534
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067160

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0101119, 0.0101346
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0101558, 0.0100719
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067199, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0101749, 0.0099136
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0066136, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0102320, 0.0098535
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0065733, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0102539, 0.0098439
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0065668, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0103099, 0.0097792
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0065234, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0104096, 0.0098016
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0065385, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0104667, 0.0097536
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0065062, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0104886, 0.0097379
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0064957, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0105445, 0.0096876
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0064619, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0096616, 0.0106519
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0064451

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0097118, 0.0105785
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0064788

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0097242, 0.0105679
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0064871

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0097728, 0.0104900
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0065198

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0098963, 0.0104921
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0065836

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0099465, 0.0104299
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0066173

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0099588, 0.0104111
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0066256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0100075, 0.0103484
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0066583

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0100423, 0.0101901
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067007

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0101046, 0.0101300
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0101211, 0.0101204
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0101829, 0.0100556
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0066882, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0102770, 0.0100781
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067033, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0103393, 0.0100301
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0066711, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0103558, 0.0100144
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0066605, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0104176, 0.0099641
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0066267, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0096512, 0.0106876
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0064381

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0097016, 0.0106142
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0064719

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0097172, 0.0106036
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0064825

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0097653, 0.0105257
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0065147

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0098859, 0.0105278
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0065766

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0099362, 0.0104656
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0066104

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0099519, 0.0104468
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0066210

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0100000, 0.0103841
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0066532

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0100355, 0.0102258
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0066962

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0100983, 0.0101657
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0101171, 0.0101561
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0101793, 0.0100913
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067121, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0102702, 0.0101138
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0103330, 0.0100658
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0066949, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0103518, 0.0100501
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0066844, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0104140, 0.0099998
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0066506, 0.0067218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
time: 0.74 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.75 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 2, lower bound: -0.0052447, upper bound: 0.0052447

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.04 + 400.05 = 403.09 seconds
