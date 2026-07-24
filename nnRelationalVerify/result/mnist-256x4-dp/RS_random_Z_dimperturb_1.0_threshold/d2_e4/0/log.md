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
Threshold: 123.8455902402


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=139, inp2_unstable=139, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=244, inp2_unstable=244, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-77.7280884, 59.2925224, -77.7280884, 59.2925224, -137.0205994, 137.0205994)
1: (-62.1903458, 55.4684753, -62.1903458, 55.4684753, -117.6588211, 117.6588211)
2: (-83.4707642, 55.5626564, -83.4707642, 55.5626564, -139.0334167, 139.0334167)
3: (-90.8407211, 46.5324059, -90.8407211, 46.5324059, -137.3731079, 137.3730927)
4: (-88.2189636, 59.1249161, -88.2189636, 59.1249161, -147.3438721, 147.3438721)
5: (-78.1039047, 53.5018120, -78.1039047, 53.5018120, -131.6056824, 131.6056824)
6: (-80.4469910, 59.0589066, -80.4469910, 59.0589066, -139.5058594, 139.5058746)
7: (-74.1075287, 66.7798309, -74.1075287, 66.7798309, -140.8873444, 140.8873444)
8: (-98.0048981, 58.9706573, -98.0048981, 58.9706573, -156.9755554, 156.9755554)
9: (-68.4725647, 69.6153107, -68.4725647, 69.6153107, -138.0878754, 138.0878601)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.31 + 15.84 = 17.16 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -123.9695598, upper bound: 123.9695598

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9084010, upper bound: 123.9084010
time: 26.92 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9084010, upper bound: 123.9084010
time: 26.89 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 53.82 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 53.82
Output dim: 6, lower bound: -123.9084010, upper bound: 123.9084010
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 53.82
Output dim: 6, lower bound: -123.9084010, upper bound: 123.9084010

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -77.7280884, 59.2925224, -77.7280884, 59.2925224, -137.0205994, 137.0205994
1: -62.1903458, 55.4684753, -62.1903458, 55.4684753, -117.6588211, 117.6588211
2: -83.4707642, 55.5626564, -83.4707642, 55.5626564, -139.0334167, 139.0334167
3: -90.8407211, 46.5324059, -90.8407211, 46.5324059, -137.3731079, 137.3730927
4: -88.2189636, 59.1249161, -88.2189636, 59.1249161, -147.3438721, 147.3438721
5: -78.1039047, 53.5018120, -78.1039047, 53.5018120, -131.6056824, 131.6056824
6: -80.4469910, 59.0589066, -80.4469910, 59.0589066, -139.5058594, 139.5058746
7: -74.1075287, 66.7798309, -74.1075287, 66.7798309, -140.8873444, 140.8873444
8: -98.0048981, 58.9706573, -98.0048981, 58.9706573, -156.9755554, 156.9755554
9: -68.4725647, 69.6153107, -68.4725647, 69.6153107, -138.0878754, 138.0878601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=139, inp2_unstable=139, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=244, inp2_unstable=244, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9056985, upper bound: 123.9056891
time: 10.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9056891, upper bound: 123.9056985
time: 10.78 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -77.7280884, 59.2925224, -77.7280884, 59.2925224, -137.0205994, 137.0205994
1: -62.1903458, 55.4684753, -62.1903458, 55.4684753, -117.6588211, 117.6588211
2: -83.4707642, 55.5626564, -83.4707642, 55.5626564, -139.0334167, 139.0334167
3: -90.8407211, 46.5324059, -90.8407211, 46.5324059, -137.3731079, 137.3730927
4: -88.2189636, 59.1249161, -88.2189636, 59.1249161, -147.3438721, 147.3438721
5: -78.1039047, 53.5018120, -78.1039047, 53.5018120, -131.6056824, 131.6056824
6: -80.4469910, 59.0589066, -80.4469910, 59.0589066, -139.5058594, 139.5058746
7: -74.1075287, 66.7798309, -74.1075287, 66.7798309, -140.8873444, 140.8873444
8: -98.0048981, 58.9706573, -98.0048981, 58.9706573, -156.9755554, 156.9755554
9: -68.4725647, 69.6153107, -68.4725647, 69.6153107, -138.0878754, 138.0878601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=139, inp2_unstable=139, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=244, inp2_unstable=244, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9030418, upper bound: 123.9030409
time: 13.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9030409, upper bound: 123.9030418
time: 12.02 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 26.89 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 26.89
Output dim: 6, lower bound: -123.9056985, upper bound: 123.9056891
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 26.89
Output dim: 6, lower bound: -123.9056891, upper bound: 123.9056985
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 26.89
Output dim: 6, lower bound: -123.9030418, upper bound: 123.9030409
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 26.89
Output dim: 6, lower bound: -123.9030409, upper bound: 123.9030418

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -77.7280884, 59.2925224, -77.7280884, 59.2925224, -137.0205994, 137.0205994
1: -62.1903458, 55.4684753, -62.1903458, 55.4684753, -117.6588211, 117.6588211
2: -83.4707642, 55.5626564, -83.4707642, 55.5626564, -139.0334167, 139.0334167
3: -90.8407211, 46.5324059, -90.8407211, 46.5324059, -137.3731079, 137.3730927
4: -88.2189636, 59.1249161, -88.2189636, 59.1249161, -147.3438721, 147.3438721
5: -78.1039047, 53.5018120, -78.1039047, 53.5018120, -131.6056824, 131.6056824
6: -80.4469910, 59.0589066, -80.4469910, 59.0589066, -139.5058594, 139.5058746
7: -74.1075287, 66.7798309, -74.1075287, 66.7798309, -140.8873444, 140.8873444
8: -98.0048981, 58.9706573, -98.0048981, 58.9706573, -156.9755554, 156.9755554
9: -68.4725647, 69.6153107, -68.4725647, 69.6153107, -138.0878754, 138.0878601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=139, inp2_unstable=139, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=244, inp2_unstable=244, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 169

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9004083, upper bound: 123.9004054
time: 9.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9004055, upper bound: 123.9004071
time: 11.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -77.7280884, 59.2925224, -77.7280884, 59.2925224, -137.0205994, 137.0205994
1: -62.1903458, 55.4684753, -62.1903458, 55.4684753, -117.6588211, 117.6588211
2: -83.4707642, 55.5626564, -83.4707642, 55.5626564, -139.0334167, 139.0334167
3: -90.8407211, 46.5324059, -90.8407211, 46.5324059, -137.3731079, 137.3730927
4: -88.2189636, 59.1249161, -88.2189636, 59.1249161, -147.3438721, 147.3438721
5: -78.1039047, 53.5018120, -78.1039047, 53.5018120, -131.6056824, 131.6056824
6: -80.4469910, 59.0589066, -80.4469910, 59.0589066, -139.5058594, 139.5058746
7: -74.1075287, 66.7798309, -74.1075287, 66.7798309, -140.8873444, 140.8873444
8: -98.0048981, 58.9706573, -98.0048981, 58.9706573, -156.9755554, 156.9755554
9: -68.4725647, 69.6153107, -68.4725647, 69.6153107, -138.0878754, 138.0878601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=139, inp2_unstable=139, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=244, inp2_unstable=244, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9037809, upper bound: 123.9037885
time: 9.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9037809, upper bound: 123.9037883
time: 14.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -77.7280884, 59.2925224, -77.7280884, 59.2925224, -137.0205994, 137.0205994
1: -62.1903458, 55.4684753, -62.1903458, 55.4684753, -117.6588211, 117.6588211
2: -83.4707642, 55.5626564, -83.4707642, 55.5626564, -139.0334167, 139.0334167
3: -90.8407211, 46.5324059, -90.8407211, 46.5324059, -137.3731079, 137.3730927
4: -88.2189636, 59.1249161, -88.2189636, 59.1249161, -147.3438721, 147.3438721
5: -78.1039047, 53.5018120, -78.1039047, 53.5018120, -131.6056824, 131.6056824
6: -80.4469910, 59.0589066, -80.4469910, 59.0589066, -139.5058594, 139.5058746
7: -74.1075287, 66.7798309, -74.1075287, 66.7798309, -140.8873444, 140.8873444
8: -98.0048981, 58.9706573, -98.0048981, 58.9706573, -156.9755554, 156.9755554
9: -68.4725647, 69.6153107, -68.4725647, 69.6153107, -138.0878754, 138.0878601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=139, inp2_unstable=139, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=244, inp2_unstable=244, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 169

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.8995423, upper bound: 123.8995346
time: 11.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.8995346, upper bound: 123.8995418
time: 8.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -77.7280884, 59.2925224, -77.7280884, 59.2925224, -137.0205994, 137.0205994
1: -62.1903458, 55.4684753, -62.1903458, 55.4684753, -117.6588211, 117.6588211
2: -83.4707642, 55.5626564, -83.4707642, 55.5626564, -139.0334167, 139.0334167
3: -90.8407211, 46.5324059, -90.8407211, 46.5324059, -137.3731079, 137.3730927
4: -88.2189636, 59.1249161, -88.2189636, 59.1249161, -147.3438721, 147.3438721
5: -78.1039047, 53.5018120, -78.1039047, 53.5018120, -131.6056824, 131.6056824
6: -80.4469910, 59.0589066, -80.4469910, 59.0589066, -139.5058594, 139.5058746
7: -74.1075287, 66.7798309, -74.1075287, 66.7798309, -140.8873444, 140.8873444
8: -98.0048981, 58.9706573, -98.0048981, 58.9706573, -156.9755554, 156.9755554
9: -68.4725647, 69.6153107, -68.4725647, 69.6153107, -138.0878754, 138.0878601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=139, inp2_unstable=139, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=244, inp2_unstable=244, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9030409, upper bound: 123.9030400
time: 13.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9030397, upper bound: 123.9030418
time: 12.23 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 27.02 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.02
Output dim: 6, lower bound: -123.9004083, upper bound: 123.9004054
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.02
Output dim: 6, lower bound: -123.9004055, upper bound: 123.9004071
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.02
Output dim: 6, lower bound: -123.9037809, upper bound: 123.9037885
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.02
Output dim: 6, lower bound: -123.9037809, upper bound: 123.9037883
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.02
Output dim: 6, lower bound: -123.8995423, upper bound: 123.8995346
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.02
Output dim: 6, lower bound: -123.8995346, upper bound: 123.8995418
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.02
Output dim: 6, lower bound: -123.9030409, upper bound: 123.9030400
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.02
Output dim: 6, lower bound: -123.9030397, upper bound: 123.9030418

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -77.7280884, 59.2925224, -77.7280884, 59.2925224, -137.0205994, 137.0205994
1: -62.1903458, 55.4684753, -62.1903458, 55.4684753, -117.6588211, 117.6588211
2: -83.4707642, 55.5626564, -83.4707642, 55.5626564, -139.0334167, 139.0334167
3: -90.8407211, 46.5324059, -90.8407211, 46.5324059, -137.3731079, 137.3730927
4: -88.2189636, 59.1249161, -88.2189636, 59.1249161, -147.3438721, 147.3438721
5: -78.1039047, 53.5018120, -78.1039047, 53.5018120, -131.6056824, 131.6056824
6: -80.4469910, 59.0589066, -80.4469910, 59.0589066, -139.5058594, 139.5058746
7: -74.1075287, 66.7798309, -74.1075287, 66.7798309, -140.8873444, 140.8873444
8: -98.0048981, 58.9706573, -98.0048981, 58.9706573, -156.9755554, 156.9755554
9: -68.4725647, 69.6153107, -68.4725647, 69.6153107, -138.0878754, 138.0878601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=139, inp2_unstable=139, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=244, inp2_unstable=244, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.8997639, upper bound: 123.8997573
time: 13.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.8997618, upper bound: 123.8997599
time: 12.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -77.7280884, 59.2925224, -77.7280884, 59.2925224, -137.0205994, 137.0205994
1: -62.1903458, 55.4684753, -62.1903458, 55.4684753, -117.6588211, 117.6588211
2: -83.4707642, 55.5626564, -83.4707642, 55.5626564, -139.0334167, 139.0334167
3: -90.8407211, 46.5324059, -90.8407211, 46.5324059, -137.3731079, 137.3730927
4: -88.2189636, 59.1249161, -88.2189636, 59.1249161, -147.3438721, 147.3438721
5: -78.1039047, 53.5018120, -78.1039047, 53.5018120, -131.6056824, 131.6056824
6: -80.4469910, 59.0589066, -80.4469910, 59.0589066, -139.5058594, 139.5058746
7: -74.1075287, 66.7798309, -74.1075287, 66.7798309, -140.8873444, 140.8873444
8: -98.0048981, 58.9706573, -98.0048981, 58.9706573, -156.9755554, 156.9755554
9: -68.4725647, 69.6153107, -68.4725647, 69.6153107, -138.0878754, 138.0878601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=139, inp2_unstable=139, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=244, inp2_unstable=244, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 139

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.8980677, upper bound: 123.8980645
time: 9.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.8980679, upper bound: 123.8980640
time: 9.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -77.7280884, 59.2925224, -77.7280884, 59.2925224, -137.0205994, 137.0205994
1: -62.1903458, 55.4684753, -62.1903458, 55.4684753, -117.6588211, 117.6588211
2: -83.4707642, 55.5626564, -83.4707642, 55.5626564, -139.0334167, 139.0334167
3: -90.8407211, 46.5324059, -90.8407211, 46.5324059, -137.3731079, 137.3730927
4: -88.2189636, 59.1249161, -88.2189636, 59.1249161, -147.3438721, 147.3438721
5: -78.1039047, 53.5018120, -78.1039047, 53.5018120, -131.6056824, 131.6056824
6: -80.4469910, 59.0589066, -80.4469910, 59.0589066, -139.5058594, 139.5058746
7: -74.1075287, 66.7798309, -74.1075287, 66.7798309, -140.8873444, 140.8873444
8: -98.0048981, 58.9706573, -98.0048981, 58.9706573, -156.9755554, 156.9755554
9: -68.4725647, 69.6153107, -68.4725647, 69.6153107, -138.0878754, 138.0878601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=139, inp2_unstable=139, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=244, inp2_unstable=244, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 60

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9012542, upper bound: 123.9012549
time: 13.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9012542, upper bound: 123.9012549
time: 12.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -77.7280884, 59.2925224, -77.7280884, 59.2925224, -137.0205994, 137.0205994
1: -62.1903458, 55.4684753, -62.1903458, 55.4684753, -117.6588211, 117.6588211
2: -83.4707642, 55.5626564, -83.4707642, 55.5626564, -139.0334167, 139.0334167
3: -90.8407211, 46.5324059, -90.8407211, 46.5324059, -137.3731079, 137.3730927
4: -88.2189636, 59.1249161, -88.2189636, 59.1249161, -147.3438721, 147.3438721
5: -78.1039047, 53.5018120, -78.1039047, 53.5018120, -131.6056824, 131.6056824
6: -80.4469910, 59.0589066, -80.4469910, 59.0589066, -139.5058594, 139.5058746
7: -74.1075287, 66.7798309, -74.1075287, 66.7798309, -140.8873444, 140.8873444
8: -98.0048981, 58.9706573, -98.0048981, 58.9706573, -156.9755554, 156.9755554
9: -68.4725647, 69.6153107, -68.4725647, 69.6153107, -138.0878754, 138.0878601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=139, inp2_unstable=139, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=244, inp2_unstable=244, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 163

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9008525, upper bound: 123.9008626
time: 14.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9008525, upper bound: 123.9008627
time: 10.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -77.7280884, 59.2925224, -77.7280884, 59.2925224, -137.0205994, 137.0205994
1: -62.1903458, 55.4684753, -62.1903458, 55.4684753, -117.6588211, 117.6588211
2: -83.4707642, 55.5626564, -83.4707642, 55.5626564, -139.0334167, 139.0334167
3: -90.8407211, 46.5324059, -90.8407211, 46.5324059, -137.3731079, 137.3730927
4: -88.2189636, 59.1249161, -88.2189636, 59.1249161, -147.3438721, 147.3438721
5: -78.1039047, 53.5018120, -78.1039047, 53.5018120, -131.6056824, 131.6056824
6: -80.4469910, 59.0589066, -80.4469910, 59.0589066, -139.5058594, 139.5058746
7: -74.1075287, 66.7798309, -74.1075287, 66.7798309, -140.8873444, 140.8873444
8: -98.0048981, 58.9706573, -98.0048981, 58.9706573, -156.9755554, 156.9755554
9: -68.4725647, 69.6153107, -68.4725647, 69.6153107, -138.0878754, 138.0878601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=139, inp2_unstable=139, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=244, inp2_unstable=244, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.8979827, upper bound: 123.8979695
time: 12.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.8979743, upper bound: 123.8979743
time: 16.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -77.7280884, 59.2925224, -77.7280884, 59.2925224, -137.0205994, 137.0205994
1: -62.1903458, 55.4684753, -62.1903458, 55.4684753, -117.6588211, 117.6588211
2: -83.4707642, 55.5626564, -83.4707642, 55.5626564, -139.0334167, 139.0334167
3: -90.8407211, 46.5324059, -90.8407211, 46.5324059, -137.3731079, 137.3730927
4: -88.2189636, 59.1249161, -88.2189636, 59.1249161, -147.3438721, 147.3438721
5: -78.1039047, 53.5018120, -78.1039047, 53.5018120, -131.6056824, 131.6056824
6: -80.4469910, 59.0589066, -80.4469910, 59.0589066, -139.5058594, 139.5058746
7: -74.1075287, 66.7798309, -74.1075287, 66.7798309, -140.8873444, 140.8873444
8: -98.0048981, 58.9706573, -98.0048981, 58.9706573, -156.9755554, 156.9755554
9: -68.4725647, 69.6153107, -68.4725647, 69.6153107, -138.0878754, 138.0878601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=139, inp2_unstable=139, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=244, inp2_unstable=244, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 60

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.8967210, upper bound: 123.8967260
time: 11.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.8967209, upper bound: 123.8967292
time: 11.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -77.7280884, 59.2925224, -77.7280884, 59.2925224, -137.0205994, 137.0205994
1: -62.1903458, 55.4684753, -62.1903458, 55.4684753, -117.6588211, 117.6588211
2: -83.4707642, 55.5626564, -83.4707642, 55.5626564, -139.0334167, 139.0334167
3: -90.8407211, 46.5324059, -90.8407211, 46.5324059, -137.3731079, 137.3730927
4: -88.2189636, 59.1249161, -88.2189636, 59.1249161, -147.3438721, 147.3438721
5: -78.1039047, 53.5018120, -78.1039047, 53.5018120, -131.6056824, 131.6056824
6: -80.4469910, 59.0589066, -80.4469910, 59.0589066, -139.5058594, 139.5058746
7: -74.1075287, 66.7798309, -74.1075287, 66.7798309, -140.8873444, 140.8873444
8: -98.0048981, 58.9706573, -98.0048981, 58.9706573, -156.9755554, 156.9755554
9: -68.4725647, 69.6153107, -68.4725647, 69.6153107, -138.0878754, 138.0878601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=139, inp2_unstable=139, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=244, inp2_unstable=244, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9030409, upper bound: 123.9030372
time: 14.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9030395, upper bound: 123.9030400
time: 10.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -77.7280884, 59.2925224, -77.7280884, 59.2925224, -137.0205994, 137.0205994
1: -62.1903458, 55.4684753, -62.1903458, 55.4684753, -117.6588211, 117.6588211
2: -83.4707642, 55.5626564, -83.4707642, 55.5626564, -139.0334167, 139.0334167
3: -90.8407211, 46.5324059, -90.8407211, 46.5324059, -137.3731079, 137.3730927
4: -88.2189636, 59.1249161, -88.2189636, 59.1249161, -147.3438721, 147.3438721
5: -78.1039047, 53.5018120, -78.1039047, 53.5018120, -131.6056824, 131.6056824
6: -80.4469910, 59.0589066, -80.4469910, 59.0589066, -139.5058594, 139.5058746
7: -74.1075287, 66.7798309, -74.1075287, 66.7798309, -140.8873444, 140.8873444
8: -98.0048981, 58.9706573, -98.0048981, 58.9706573, -156.9755554, 156.9755554
9: -68.4725647, 69.6153107, -68.4725647, 69.6153107, -138.0878754, 138.0878601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=139, inp2_unstable=139, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=244, inp2_unstable=244, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 60

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9007353, upper bound: 123.9007408
time: 11.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9007375, upper bound: 123.9007407
time: 9.39 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 21.65 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.65
Output dim: 6, lower bound: -123.8997639, upper bound: 123.8997573
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.65
Output dim: 6, lower bound: -123.8997618, upper bound: 123.8997599
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.65
Output dim: 6, lower bound: -123.8980677, upper bound: 123.8980645
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.65
Output dim: 6, lower bound: -123.8980679, upper bound: 123.8980640
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.65
Output dim: 6, lower bound: -123.9012542, upper bound: 123.9012549
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.65
Output dim: 6, lower bound: -123.9012542, upper bound: 123.9012549
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.65
Output dim: 6, lower bound: -123.9008525, upper bound: 123.9008626
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.65
Output dim: 6, lower bound: -123.9008525, upper bound: 123.9008627
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.65
Output dim: 6, lower bound: -123.8979827, upper bound: 123.8979695
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.65
Output dim: 6, lower bound: -123.8979743, upper bound: 123.8979743
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.65
Output dim: 6, lower bound: -123.8967210, upper bound: 123.8967260
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.65
Output dim: 6, lower bound: -123.8967209, upper bound: 123.8967292
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.65
Output dim: 6, lower bound: -123.9030409, upper bound: 123.9030372
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.65
Output dim: 6, lower bound: -123.9030395, upper bound: 123.9030400
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.65
Output dim: 6, lower bound: -123.9007353, upper bound: 123.9007408
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.65
Output dim: 6, lower bound: -123.9007375, upper bound: 123.9007407

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -77.7280884, 59.2925224, -77.7280884, 59.2925224, -137.0205994, 137.0205994
1: -62.1903458, 55.4684753, -62.1903458, 55.4684753, -117.6588211, 117.6588211
2: -83.4707642, 55.5626564, -83.4707642, 55.5626564, -139.0334167, 139.0334167
3: -90.8407211, 46.5324059, -90.8407211, 46.5324059, -137.3731079, 137.3730927
4: -88.2189636, 59.1249161, -88.2189636, 59.1249161, -147.3438721, 147.3438721
5: -78.1039047, 53.5018120, -78.1039047, 53.5018120, -131.6056824, 131.6056824
6: -80.4469910, 59.0589066, -80.4469910, 59.0589066, -139.5058594, 139.5058746
7: -74.1075287, 66.7798309, -74.1075287, 66.7798309, -140.8873444, 140.8873444
8: -98.0048981, 58.9706573, -98.0048981, 58.9706573, -156.9755554, 156.9755554
9: -68.4725647, 69.6153107, -68.4725647, 69.6153107, -138.0878754, 138.0878601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=139, inp2_unstable=139, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=244, inp2_unstable=244, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.8904145, upper bound: 123.8904175
time: 10.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.8904145, upper bound: 123.8904175
time: 12.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -77.7280884, 59.2925224, -77.7280884, 59.2925224, -137.0205994, 137.0205994
1: -62.1903458, 55.4684753, -62.1903458, 55.4684753, -117.6588211, 117.6588211
2: -83.4707642, 55.5626564, -83.4707642, 55.5626564, -139.0334167, 139.0334167
3: -90.8407211, 46.5324059, -90.8407211, 46.5324059, -137.3731079, 137.3730927
4: -88.2189636, 59.1249161, -88.2189636, 59.1249161, -147.3438721, 147.3438721
5: -78.1039047, 53.5018120, -78.1039047, 53.5018120, -131.6056824, 131.6056824
6: -80.4469910, 59.0589066, -80.4469910, 59.0589066, -139.5058594, 139.5058746
7: -74.1075287, 66.7798309, -74.1075287, 66.7798309, -140.8873444, 140.8873444
8: -98.0048981, 58.9706573, -98.0048981, 58.9706573, -156.9755554, 156.9755554
9: -68.4725647, 69.6153107, -68.4725647, 69.6153107, -138.0878754, 138.0878601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=139, inp2_unstable=139, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=244, inp2_unstable=244, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.8997618, upper bound: 123.8997593
time: 11.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.8997617, upper bound: 123.8997599
time: 14.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -77.7280884, 59.2925224, -77.7280884, 59.2925224, -137.0205994, 137.0205994
1: -62.1903458, 55.4684753, -62.1903458, 55.4684753, -117.6588211, 117.6588211
2: -83.4707642, 55.5626564, -83.4707642, 55.5626564, -139.0334167, 139.0334167
3: -90.8407211, 46.5324059, -90.8407211, 46.5324059, -137.3731079, 137.3730927
4: -88.2189636, 59.1249161, -88.2189636, 59.1249161, -147.3438721, 147.3438721
5: -78.1039047, 53.5018120, -78.1039047, 53.5018120, -131.6056824, 131.6056824
6: -80.4469910, 59.0589066, -80.4469910, 59.0589066, -139.5058594, 139.5058746
7: -74.1075287, 66.7798309, -74.1075287, 66.7798309, -140.8873444, 140.8873444
8: -98.0048981, 58.9706573, -98.0048981, 58.9706573, -156.9755554, 156.9755554
9: -68.4725647, 69.6153107, -68.4725647, 69.6153107, -138.0878754, 138.0878601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=139, inp2_unstable=139, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=244, inp2_unstable=244, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.8980674, upper bound: 123.8980645
time: 11.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.8980614, upper bound: 123.8980629
time: 18.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -77.7280884, 59.2925224, -77.7280884, 59.2925224, -137.0205994, 137.0205994
1: -62.1903458, 55.4684753, -62.1903458, 55.4684753, -117.6588211, 117.6588211
2: -83.4707642, 55.5626564, -83.4707642, 55.5626564, -139.0334167, 139.0334167
3: -90.8407211, 46.5324059, -90.8407211, 46.5324059, -137.3731079, 137.3730927
4: -88.2189636, 59.1249161, -88.2189636, 59.1249161, -147.3438721, 147.3438721
5: -78.1039047, 53.5018120, -78.1039047, 53.5018120, -131.6056824, 131.6056824
6: -80.4469910, 59.0589066, -80.4469910, 59.0589066, -139.5058594, 139.5058746
7: -74.1075287, 66.7798309, -74.1075287, 66.7798309, -140.8873444, 140.8873444
8: -98.0048981, 58.9706573, -98.0048981, 58.9706573, -156.9755554, 156.9755554
9: -68.4725647, 69.6153107, -68.4725647, 69.6153107, -138.0878754, 138.0878601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=139, inp2_unstable=139, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=244, inp2_unstable=244, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 130

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.8980679, upper bound: 123.8980622
time: 14.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.8980675, upper bound: 123.8980640
time: 12.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -77.7280884, 59.2925224, -77.7280884, 59.2925224, -137.0205994, 137.0205994
1: -62.1903458, 55.4684753, -62.1903458, 55.4684753, -117.6588211, 117.6588211
2: -83.4707642, 55.5626564, -83.4707642, 55.5626564, -139.0334167, 139.0334167
3: -90.8407211, 46.5324059, -90.8407211, 46.5324059, -137.3731079, 137.3730927
4: -88.2189636, 59.1249161, -88.2189636, 59.1249161, -147.3438721, 147.3438721
5: -78.1039047, 53.5018120, -78.1039047, 53.5018120, -131.6056824, 131.6056824
6: -80.4469910, 59.0589066, -80.4469910, 59.0589066, -139.5058594, 139.5058746
7: -74.1075287, 66.7798309, -74.1075287, 66.7798309, -140.8873444, 140.8873444
8: -98.0048981, 58.9706573, -98.0048981, 58.9706573, -156.9755554, 156.9755554
9: -68.4725647, 69.6153107, -68.4725647, 69.6153107, -138.0878754, 138.0878601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=139, inp2_unstable=139, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=244, inp2_unstable=244, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 139

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9006663, upper bound: 123.9006679
time: 12.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9006659, upper bound: 123.9006675
time: 11.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -77.7280884, 59.2925224, -77.7280884, 59.2925224, -137.0205994, 137.0205994
1: -62.1903458, 55.4684753, -62.1903458, 55.4684753, -117.6588211, 117.6588211
2: -83.4707642, 55.5626564, -83.4707642, 55.5626564, -139.0334167, 139.0334167
3: -90.8407211, 46.5324059, -90.8407211, 46.5324059, -137.3731079, 137.3730927
4: -88.2189636, 59.1249161, -88.2189636, 59.1249161, -147.3438721, 147.3438721
5: -78.1039047, 53.5018120, -78.1039047, 53.5018120, -131.6056824, 131.6056824
6: -80.4469910, 59.0589066, -80.4469910, 59.0589066, -139.5058594, 139.5058746
7: -74.1075287, 66.7798309, -74.1075287, 66.7798309, -140.8873444, 140.8873444
8: -98.0048981, 58.9706573, -98.0048981, 58.9706573, -156.9755554, 156.9755554
9: -68.4725647, 69.6153107, -68.4725647, 69.6153107, -138.0878754, 138.0878601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=139, inp2_unstable=139, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=244, inp2_unstable=244, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9012542, upper bound: 123.9012540
time: 11.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9012539, upper bound: 123.9012549
time: 10.02 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 22.83 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 6, lower bound: -123.8904145, upper bound: 123.8904175
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 6, lower bound: -123.8904145, upper bound: 123.8904175
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 6, lower bound: -123.8997618, upper bound: 123.8997593
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 6, lower bound: -123.8997617, upper bound: 123.8997599
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 6, lower bound: -123.8980674, upper bound: 123.8980645
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 6, lower bound: -123.8980614, upper bound: 123.8980629
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 6, lower bound: -123.8980679, upper bound: 123.8980622
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 6, lower bound: -123.8980675, upper bound: 123.8980640
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 6, lower bound: -123.9006663, upper bound: 123.9006679
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 6, lower bound: -123.9006659, upper bound: 123.9006675
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 6, lower bound: -123.9012542, upper bound: 123.9012540
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 6, lower bound: -123.9012539, upper bound: 123.9012549
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.83
Output dim: 6, lower bound: -123.9008525, upper bound: 123.9008626
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.83
Output dim: 6, lower bound: -123.9008525, upper bound: 123.9008627
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.83
Output dim: 6, lower bound: -123.8979827, upper bound: 123.8979695
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.83
Output dim: 6, lower bound: -123.8979743, upper bound: 123.8979743
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.83
Output dim: 6, lower bound: -123.8967210, upper bound: 123.8967260
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.83
Output dim: 6, lower bound: -123.8967209, upper bound: 123.8967292
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.83
Output dim: 6, lower bound: -123.9030409, upper bound: 123.9030372
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.83
Output dim: 6, lower bound: -123.9030395, upper bound: 123.9030400
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.83
Output dim: 6, lower bound: -123.9007353, upper bound: 123.9007408
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.83
Output dim: 6, lower bound: -123.9007375, upper bound: 123.9007407

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 17.16 + 597.23 = 614.38 seconds
