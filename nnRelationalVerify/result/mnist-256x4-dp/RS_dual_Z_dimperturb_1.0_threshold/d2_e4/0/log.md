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
execution time: IAR + RelationalAnalysis = 1.34 + 15.87 = 17.22 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -123.9695598, upper bound: 123.9695598

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9695598, upper bound: 123.9695598
time: 13.63 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9695598, upper bound: 123.9695598
time: 13.18 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 26.94 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 26.94
Output dim: 6, lower bound: -123.9695598, upper bound: 123.9695598
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 26.94
Output dim: 6, lower bound: -123.9695598, upper bound: 123.9695598

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9695598, upper bound: 123.9695431
time: 15.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9695431, upper bound: 123.9695598
time: 14.80 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9695598, upper bound: 123.9695431
time: 15.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9695431, upper bound: 123.9695598
time: 14.49 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 30.99 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 30.99
Output dim: 6, lower bound: -123.9695598, upper bound: 123.9695431
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 30.99
Output dim: 6, lower bound: -123.9695431, upper bound: 123.9695598
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 30.99
Output dim: 6, lower bound: -123.9695598, upper bound: 123.9695431
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 30.99
Output dim: 6, lower bound: -123.9695431, upper bound: 123.9695598

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9611001, upper bound: 123.9610854
time: 12.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9611001, upper bound: 123.9610854
time: 12.01 seconds

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
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9610854, upper bound: 123.9611001
time: 11.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9610854, upper bound: 123.9611001
time: 16.10 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9611001, upper bound: 123.9610854
time: 11.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9611001, upper bound: 123.9610854
time: 12.73 seconds

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
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9610854, upper bound: 123.9611001
time: 12.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9610854, upper bound: 123.9611001
time: 12.71 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 26.42 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.42
Output dim: 6, lower bound: -123.9611001, upper bound: 123.9610854
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.42
Output dim: 6, lower bound: -123.9611001, upper bound: 123.9610854
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.42
Output dim: 6, lower bound: -123.9610854, upper bound: 123.9611001
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.42
Output dim: 6, lower bound: -123.9610854, upper bound: 123.9611001
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.42
Output dim: 6, lower bound: -123.9611001, upper bound: 123.9610854
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.42
Output dim: 6, lower bound: -123.9611001, upper bound: 123.9610854
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.42
Output dim: 6, lower bound: -123.9610854, upper bound: 123.9611001
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.42
Output dim: 6, lower bound: -123.9610854, upper bound: 123.9611001

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9611001, upper bound: 123.9610854
time: 14.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9610983, upper bound: 123.9610853
time: 14.92 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9611001, upper bound: 123.9610854
time: 14.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9610983, upper bound: 123.9610853
time: 13.99 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9610853, upper bound: 123.9610983
time: 14.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9610854, upper bound: 123.9611001
time: 12.68 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9610853, upper bound: 123.9610983
time: 11.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9610854, upper bound: 123.9611001
time: 11.82 seconds

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
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9611001, upper bound: 123.9610854
time: 15.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9610983, upper bound: 123.9610853
time: 10.46 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9611001, upper bound: 123.9610854
time: 13.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9610983, upper bound: 123.9610853
time: 18.11 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9610853, upper bound: 123.9610983
time: 12.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9610854, upper bound: 123.9611001
time: 11.18 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9610853, upper bound: 123.9610983
time: 12.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9610854, upper bound: 123.9611001
time: 14.47 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 28.20 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.20
Output dim: 6, lower bound: -123.9611001, upper bound: 123.9610854
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.20
Output dim: 6, lower bound: -123.9610983, upper bound: 123.9610853
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.20
Output dim: 6, lower bound: -123.9611001, upper bound: 123.9610854
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.20
Output dim: 6, lower bound: -123.9610983, upper bound: 123.9610853
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.20
Output dim: 6, lower bound: -123.9610853, upper bound: 123.9610983
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.20
Output dim: 6, lower bound: -123.9610854, upper bound: 123.9611001
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.20
Output dim: 6, lower bound: -123.9610853, upper bound: 123.9610983
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.20
Output dim: 6, lower bound: -123.9610854, upper bound: 123.9611001
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.20
Output dim: 6, lower bound: -123.9611001, upper bound: 123.9610854
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.20
Output dim: 6, lower bound: -123.9610983, upper bound: 123.9610853
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.20
Output dim: 6, lower bound: -123.9611001, upper bound: 123.9610854
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.20
Output dim: 6, lower bound: -123.9610983, upper bound: 123.9610853
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.20
Output dim: 6, lower bound: -123.9610853, upper bound: 123.9610983
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.20
Output dim: 6, lower bound: -123.9610854, upper bound: 123.9611001
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.20
Output dim: 6, lower bound: -123.9610853, upper bound: 123.9610983
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.20
Output dim: 6, lower bound: -123.9610854, upper bound: 123.9611001

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
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9601287, upper bound: 123.9601041
time: 10.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9601304, upper bound: 123.9601042
time: 14.02 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9601202, upper bound: 123.9601050
time: 14.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9601212, upper bound: 123.9601051
time: 11.85 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9601287, upper bound: 123.9601041
time: 11.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9601304, upper bound: 123.9601042
time: 14.06 seconds

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
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9601042, upper bound: 123.9601050
time: 14.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9601041, upper bound: 123.9601051
time: 14.79 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9601051, upper bound: 123.9601212
time: 13.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9601050, upper bound: 123.9601202
time: 12.58 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9601042, upper bound: 123.9601304
time: 10.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9601041, upper bound: 123.9601287
time: 16.07 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 28.44 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.44
Output dim: 6, lower bound: -123.9601287, upper bound: 123.9601041
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.44
Output dim: 6, lower bound: -123.9601304, upper bound: 123.9601042
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.44
Output dim: 6, lower bound: -123.9601202, upper bound: 123.9601050
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.44
Output dim: 6, lower bound: -123.9601212, upper bound: 123.9601051
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.44
Output dim: 6, lower bound: -123.9601287, upper bound: 123.9601041
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.44
Output dim: 6, lower bound: -123.9601304, upper bound: 123.9601042
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.44
Output dim: 6, lower bound: -123.9601042, upper bound: 123.9601050
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.44
Output dim: 6, lower bound: -123.9601041, upper bound: 123.9601051
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.44
Output dim: 6, lower bound: -123.9601051, upper bound: 123.9601212
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.44
Output dim: 6, lower bound: -123.9601050, upper bound: 123.9601202
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.44
Output dim: 6, lower bound: -123.9601042, upper bound: 123.9601304
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.44
Output dim: 6, lower bound: -123.9601041, upper bound: 123.9601287
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.44
Output dim: 6, lower bound: -123.9610853, upper bound: 123.9610983
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.44
Output dim: 6, lower bound: -123.9610854, upper bound: 123.9611001
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.44
Output dim: 6, lower bound: -123.9611001, upper bound: 123.9610854
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.44
Output dim: 6, lower bound: -123.9610983, upper bound: 123.9610853
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.44
Output dim: 6, lower bound: -123.9611001, upper bound: 123.9610854
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.44
Output dim: 6, lower bound: -123.9610983, upper bound: 123.9610853
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.44
Output dim: 6, lower bound: -123.9610853, upper bound: 123.9610983
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.44
Output dim: 6, lower bound: -123.9610854, upper bound: 123.9611001
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.44
Output dim: 6, lower bound: -123.9610853, upper bound: 123.9610983
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.44
Output dim: 6, lower bound: -123.9610854, upper bound: 123.9611001

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 17.22 + 589.55 = 606.77 seconds
