## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00640413


## IAR start

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
execution time: IAR + RelationalAnalysis = 0.98 + 1.80 = 2.79 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0071157, upper bound: 0.0071157

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0070192, upper bound: 0.0070192
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0070192, upper bound: 0.0070192
time: 0.91 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.88 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.88
Output dim: 2, lower bound: -0.0070192, upper bound: 0.0070192
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.88
Output dim: 2, lower bound: -0.0070192, upper bound: 0.0070192

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0119609, 0.0118880
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 90

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0069214, upper bound: 0.0069214
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0069214, upper bound: 0.0069214
time: 1.06 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0118880, 0.0119609
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0070192, upper bound: 0.0070192
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0070192, upper bound: 0.0070192
time: 0.99 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.85 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.85
Output dim: 2, lower bound: -0.0069214, upper bound: 0.0069214
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.85
Output dim: 2, lower bound: -0.0069214, upper bound: 0.0069214
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.85
Output dim: 2, lower bound: -0.0070192, upper bound: 0.0070192
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.85
Output dim: 2, lower bound: -0.0070192, upper bound: 0.0070192

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0116666, 0.0116853
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0064038, upper bound: 0.0064038
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0064038, upper bound: 0.0064038
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0119609, 0.0115937
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0069214, upper bound: 0.0069214
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0069214, upper bound: 0.0069214
time: 1.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0104448, 0.0108848
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068270, upper bound: 0.0068270
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068270, upper bound: 0.0068270
time: 1.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0108291, 0.0105177
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0070192, upper bound: 0.0070192
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0070192, upper bound: 0.0070192
time: 1.03 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.96 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 2.96
Output dim: 2, lower bound: -0.0064038, upper bound: 0.0064038
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 2.96
Output dim: 2, lower bound: -0.0064038, upper bound: 0.0064038
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 2, lower bound: -0.0069214, upper bound: 0.0069214
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 2, lower bound: -0.0069214, upper bound: 0.0069214
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 2, lower bound: -0.0068270, upper bound: 0.0068270
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 2, lower bound: -0.0068270, upper bound: 0.0068270
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 2, lower bound: -0.0070192, upper bound: 0.0070192
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 2, lower bound: -0.0070192, upper bound: 0.0070192

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0105177, 0.0105611
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 103

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0069214, upper bound: 0.0069214
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0069214, upper bound: 0.0069214
time: 1.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0108848, 0.0101768
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0066368, 0.0067218

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0064038, upper bound: 0.0064038
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0064038, upper bound: 0.0064038
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0104408, 0.0108809
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0061445, upper bound: 0.0061445
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0061445, upper bound: 0.0061445
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0104448, 0.0108808
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067306, upper bound: 0.0067306
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067306, upper bound: 0.0067306
time: 1.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0104412, 0.0101794
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0061094, upper bound: 0.0061094
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0061094, upper bound: 0.0061094
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0105039, 0.0101297
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0066941, 0.0067218

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 90

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0069214, upper bound: 0.0069214
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0069214, upper bound: 0.0069214
time: 1.30 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.68 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 2, lower bound: -0.0069214, upper bound: 0.0069214
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 2, lower bound: -0.0069214, upper bound: 0.0069214
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.68
Output dim: 2, lower bound: -0.0064038, upper bound: 0.0064038
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.68
Output dim: 2, lower bound: -0.0064038, upper bound: 0.0064038
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.68
Output dim: 2, lower bound: -0.0061445, upper bound: 0.0061445
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.68
Output dim: 2, lower bound: -0.0061445, upper bound: 0.0061445
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 2, lower bound: -0.0067306, upper bound: 0.0067306
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 2, lower bound: -0.0067306, upper bound: 0.0067306
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.68
Output dim: 2, lower bound: -0.0061094, upper bound: 0.0061094
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.68
Output dim: 2, lower bound: -0.0061094, upper bound: 0.0061094
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 2, lower bound: -0.0069214, upper bound: 0.0069214
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 2, lower bound: -0.0069214, upper bound: 0.0069214

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0102145, 0.0103478
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0064038, upper bound: 0.0064038
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0064038, upper bound: 0.0064038
time: 1.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0102760, 0.0102663
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067130, upper bound: 0.0067130
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067130, upper bound: 0.0067130
time: 1.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0096858, 0.0100593
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0066879, 0.0064370

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065070, upper bound: 0.0065070
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065070, upper bound: 0.0065070
time: 1.11 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0096234, 0.0100928
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067104, 0.0063951

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0060409, upper bound: 0.0060409
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0060409, upper bound: 0.0060409
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0102491, 0.0099310
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0065790, 0.0067218

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068026, upper bound: 0.0068026
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068026, upper bound: 0.0068026
time: 1.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0105039, 0.0098750
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0065414, 0.0067218

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066908, upper bound: 0.0066908
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066908, upper bound: 0.0066908
time: 1.19 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.37 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.37
Output dim: 2, lower bound: -0.0064038, upper bound: 0.0064038
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.37
Output dim: 2, lower bound: -0.0064038, upper bound: 0.0064038
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 2, lower bound: -0.0067130, upper bound: 0.0067130
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 2, lower bound: -0.0067130, upper bound: 0.0067130
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 2, lower bound: -0.0065070, upper bound: 0.0065070
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 2, lower bound: -0.0065070, upper bound: 0.0065070
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.37
Output dim: 2, lower bound: -0.0060409, upper bound: 0.0060409
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.37
Output dim: 2, lower bound: -0.0060409, upper bound: 0.0060409
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 2, lower bound: -0.0068026, upper bound: 0.0068026
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 2, lower bound: -0.0068026, upper bound: 0.0068026
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 2, lower bound: -0.0066908, upper bound: 0.0066908
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 2, lower bound: -0.0066908, upper bound: 0.0066908

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0101401, 0.0099640
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0066307, 0.0067218

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066123, upper bound: 0.0066123
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066123, upper bound: 0.0066123
time: 1.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0099736, 0.0102663
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0066158

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067130, upper bound: 0.0067130
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067130, upper bound: 0.0067130
time: 1.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0095282, 0.0097536
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0065074, 0.0063559

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 90

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0063642, upper bound: 0.0063642
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0063642, upper bound: 0.0063642
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0093801, 0.0100593
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0066879, 0.0062564

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0057949, upper bound: 0.0057949
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0057949, upper bound: 0.0057949
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0102215, 0.0098970
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0065554, 0.0067218

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066981, upper bound: 0.0066981
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066981, upper bound: 0.0066981
time: 1.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0102152, 0.0099310
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0065790, 0.0067218

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0053279, upper bound: 0.0053279
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0053279, upper bound: 0.0053279
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0105000, 0.0098718
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0065393, 0.0067218

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065909, upper bound: 0.0065909
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065909, upper bound: 0.0065909
time: 1.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0105039, 0.0098710
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0065387, 0.0067218

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052986, upper bound: 0.0052986
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052986, upper bound: 0.0052986
time: 0.81 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.60 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 2, lower bound: -0.0066123, upper bound: 0.0066123
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 2, lower bound: -0.0066123, upper bound: 0.0066123
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 2, lower bound: -0.0067130, upper bound: 0.0067130
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 2, lower bound: -0.0067130, upper bound: 0.0067130
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 2, lower bound: -0.0063642, upper bound: 0.0063642
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 2, lower bound: -0.0063642, upper bound: 0.0063642
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 2, lower bound: -0.0057949, upper bound: 0.0057949
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 2, lower bound: -0.0057949, upper bound: 0.0057949
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 2, lower bound: -0.0066981, upper bound: 0.0066981
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 2, lower bound: -0.0066981, upper bound: 0.0066981
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 2, lower bound: -0.0053279, upper bound: 0.0053279
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 2, lower bound: -0.0053279, upper bound: 0.0053279
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 2, lower bound: -0.0065909, upper bound: 0.0065909
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 2, lower bound: -0.0065909, upper bound: 0.0065909
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 2, lower bound: -0.0052986, upper bound: 0.0052986
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 2, lower bound: -0.0052986, upper bound: 0.0052986

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0093573, 0.0091155
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0062149, 0.0063609

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0060789, upper bound: 0.0060789
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0060789, upper bound: 0.0060789
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0093070, 0.0091489
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0062372, 0.0063271

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0060789, upper bound: 0.0060789
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0060789, upper bound: 0.0060789
time: 1.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0095947, 0.0099664
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0064835

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066123, upper bound: 0.0066123
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066123, upper bound: 0.0066123
time: 1.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0096348, 0.0099036
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0066818, 0.0065105

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0056657, upper bound: 0.0056657
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0056657, upper bound: 0.0056657
time: 0.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0094055, 0.0090325
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0061575, 0.0064079

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064945, upper bound: 0.0064945
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064945, upper bound: 0.0064945
time: 1.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0093570, 0.0090805
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0061897, 0.0063753

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052081, upper bound: 0.0052081
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0052081, upper bound: 0.0052081
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0104724, 0.0098378
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0065157, 0.0067218

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064945, upper bound: 0.0064945
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064945, upper bound: 0.0064945
time: 1.11 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0104660, 0.0098718
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0065393, 0.0067218

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0060240, upper bound: 0.0060240
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0060240, upper bound: 0.0060240
time: 0.87 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.62 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 2, lower bound: -0.0060789, upper bound: 0.0060789
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 2, lower bound: -0.0060789, upper bound: 0.0060789
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 2, lower bound: -0.0060789, upper bound: 0.0060789
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 2, lower bound: -0.0060789, upper bound: 0.0060789
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.62
Output dim: 2, lower bound: -0.0066123, upper bound: 0.0066123
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.62
Output dim: 2, lower bound: -0.0066123, upper bound: 0.0066123
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 2, lower bound: -0.0056657, upper bound: 0.0056657
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 2, lower bound: -0.0056657, upper bound: 0.0056657
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.62
Output dim: 2, lower bound: -0.0064945, upper bound: 0.0064945
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.62
Output dim: 2, lower bound: -0.0064945, upper bound: 0.0064945
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 2, lower bound: -0.0052081, upper bound: 0.0052081
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 2, lower bound: -0.0052081, upper bound: 0.0052081
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.62
Output dim: 2, lower bound: -0.0064945, upper bound: 0.0064945
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.62
Output dim: 2, lower bound: -0.0064945, upper bound: 0.0064945
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 2, lower bound: -0.0060240, upper bound: 0.0060240
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 2, lower bound: -0.0060240, upper bound: 0.0060240

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0087829, 0.0090933
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0063231, 0.0061277

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0054188, upper bound: 0.0054188
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0054188, upper bound: 0.0054188
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0087275, 0.0091428
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0063563, 0.0060905

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0055813, upper bound: 0.0055813
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0055813, upper bound: 0.0055813
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0094012, 0.0090291
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0061552, 0.0064051

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062731, upper bound: 0.0062731
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062731, upper bound: 0.0062731
time: 1.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0094055, 0.0090283
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0061547, 0.0064079

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 103

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064945, upper bound: 0.0064945
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064945, upper bound: 0.0064945
time: 1.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0096723, 0.0089730
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0061176, 0.0065681

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 103

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064945, upper bound: 0.0064945
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064945, upper bound: 0.0064945
time: 1.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0096238, 0.0090281
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0061545, 0.0065356

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0059195, upper bound: 0.0059195
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0059195, upper bound: 0.0059195
time: 1.09 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 5.11 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.11
Output dim: 2, lower bound: -0.0054188, upper bound: 0.0054188
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.11
Output dim: 2, lower bound: -0.0054188, upper bound: 0.0054188
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.11
Output dim: 2, lower bound: -0.0055813, upper bound: 0.0055813
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.11
Output dim: 2, lower bound: -0.0055813, upper bound: 0.0055813
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.11
Output dim: 2, lower bound: -0.0062731, upper bound: 0.0062731
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.11
Output dim: 2, lower bound: -0.0062731, upper bound: 0.0062731
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 5.11
Output dim: 2, lower bound: -0.0064945, upper bound: 0.0064945
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 5.11
Output dim: 2, lower bound: -0.0064945, upper bound: 0.0064945
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 5.11
Output dim: 2, lower bound: -0.0064945, upper bound: 0.0064945
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 5.11
Output dim: 2, lower bound: -0.0064945, upper bound: 0.0064945
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.11
Output dim: 2, lower bound: -0.0059195, upper bound: 0.0059195
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.11
Output dim: 2, lower bound: -0.0059195, upper bound: 0.0059195

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0091138, 0.0088035
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0061277, 0.0063360

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062731, upper bound: 0.0062731
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062731, upper bound: 0.0062731
time: 1.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0091888, 0.0087368
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0060829, 0.0063864

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062731, upper bound: 0.0062731
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062731, upper bound: 0.0062731
time: 1.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0093692, 0.0087436
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0060875, 0.0064942

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062731, upper bound: 0.0062731
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062731, upper bound: 0.0062731
time: 1.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764
1: -0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001
2: 0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567
3: 0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0094442, 0.0086816
4: -0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686
5: 0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359
6: 0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770
7: -0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623
8: 0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190
9: 0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0060458, 0.0065446

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062731, upper bound: 0.0062731
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062731, upper bound: 0.0062731
time: 1.34 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 5.48 seconds
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 5.48
Output dim: 2, lower bound: -0.0062731, upper bound: 0.0062731
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 5.48
Output dim: 2, lower bound: -0.0062731, upper bound: 0.0062731
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 5.48
Output dim: 2, lower bound: -0.0062731, upper bound: 0.0062731
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 5.48
Output dim: 2, lower bound: -0.0062731, upper bound: 0.0062731
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 5.48
Output dim: 2, lower bound: -0.0062731, upper bound: 0.0062731
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 5.48
Output dim: 2, lower bound: -0.0062731, upper bound: 0.0062731
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 5.48
Output dim: 2, lower bound: -0.0062731, upper bound: 0.0062731
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 5.48
Output dim: 2, lower bound: -0.0062731, upper bound: 0.0062731

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.79 + 155.43 = 158.22 seconds
