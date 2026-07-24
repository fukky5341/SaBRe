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
execution time: IAR + RelationalAnalysis = 1.31 + 15.79 = 17.10 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -123.9695598, upper bound: 123.9695598

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9565599, upper bound: 123.9557317
time: 17.49 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9695598, upper bound: 123.9695598
time: 11.79 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 29.42 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 29.42
Output dim: 6, lower bound: -123.9565599, upper bound: 123.9557317
IS_B2, status: Status.UNKNOWN, split count: 1, time: 29.42
Output dim: 6, lower bound: -123.9695598, upper bound: 123.9695598

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -76.9552612, 58.7006149, -87.1539993, 66.6915970, -143.6468506, 145.8545990
1: -61.5684891, 54.9223137, -70.0167313, 62.1645737, -123.7330627, 124.9390411
2: -82.6412735, 55.0222321, -93.8220749, 62.3700256, -145.0112915, 148.8442993
3: -89.9290161, 46.0759888, -101.8941803, 52.2596817, -142.1886749, 147.9701538
4: -87.3456421, 58.5308266, -98.6428299, 66.6323776, -153.9779968, 157.1736450
5: -77.3309555, 52.9631424, -87.3176727, 60.3483047, -137.6792603, 140.2808075
6: -79.6630173, 58.4550133, -89.5108643, 67.0579681, -146.7209778, 147.9658813
7: -73.3652267, 66.1186142, -83.4477386, 74.7965546, -148.1617432, 149.5663452
8: -97.0417633, 58.3926506, -109.5759201, 66.5045166, -163.5462799, 167.9685669
9: -67.7875290, 68.9262314, -77.0091095, 78.1287537, -145.9162750, 145.9353333

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=63, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=136, inp2_unstable=140, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=242, inp2_unstable=247, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9478311, upper bound: 123.9479291
time: 18.56 seconds

## Relational analysis of IS_B1_B2

### Relational analysis result of IS_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9463311, upper bound: 123.9453535
time: 17.02 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -77.7280884, 59.2925224, -77.5312042, 59.1417313, -136.8698120, 136.8237152
1: -62.1903458, 55.4684753, -62.0316124, 55.3291054, -117.5194473, 117.5000916
2: -83.4707642, 55.5626564, -83.2591400, 55.4248352, -138.8955841, 138.8217926
3: -90.8407211, 46.5324059, -90.6080780, 46.4160271, -137.2567444, 137.1404724
4: -88.2189636, 59.1249161, -87.9964905, 58.9731598, -147.1921234, 147.1213989
5: -78.1039047, 53.5018120, -77.9071960, 53.3647423, -131.4686127, 131.4089813
6: -80.4469910, 59.0589066, -80.2473297, 58.9047890, -139.3517761, 139.3062286
7: -74.1075287, 66.7798309, -73.9184189, 66.6112366, -140.7187500, 140.6982422
8: -98.0048981, 58.9706573, -97.7594681, 58.8231773, -156.8280640, 156.7301331
9: -68.4725647, 69.6153107, -68.2982254, 69.4397278, -137.9122772, 137.9135437

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=63, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=139, inp2_unstable=138, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=244, inp2_unstable=243, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9627270, upper bound: 123.9621052
time: 16.67 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9616904, upper bound: 123.9616904
time: 13.48 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 31.50 seconds
IS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 31.50
Output dim: 6, lower bound: -123.9478311, upper bound: 123.9479291
IS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 31.50
Output dim: 6, lower bound: -123.9463311, upper bound: 123.9453535
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 31.50
Output dim: 6, lower bound: -123.9627270, upper bound: 123.9621052
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 31.50
Output dim: 6, lower bound: -123.9616904, upper bound: 123.9616904

## BFS IS instance: IS_B1_B1

### Backsubstitution after applying IS history:
0: -73.4650497, 55.9066391, -75.6133728, 57.4238091, -130.8888397, 131.5199890
1: -58.6210365, 52.4406815, -60.2386093, 53.9670715, -112.5881042, 112.6792755
2: -78.7994156, 52.4970322, -81.0944138, 53.9835625, -132.7829285, 133.5914459
3: -85.8363190, 43.9366074, -88.3493271, 45.1785431, -131.0148621, 132.2859192
4: -83.6091995, 55.6633644, -86.2837524, 57.1157951, -140.7249908, 141.9471130
5: -73.9780884, 50.3339996, -76.2233124, 51.6275253, -125.6056137, 126.5573120
6: -76.4645081, 55.3295631, -78.9345245, 56.7054672, -133.1699829, 134.2640839
7: -69.8828430, 63.1693573, -71.9072495, 65.0277786, -134.9106140, 135.0765839
8: -92.8428879, 55.4825211, -95.6765747, 56.8456573, -149.6885376, 151.1590881
9: -64.6142349, 65.7699585, -66.4788208, 67.6743927, -132.2886353, 132.2487793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=62, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=136, inp2_unstable=138, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=237, inp2_unstable=238, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 42

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_B1_B1_A1

### Relational analysis result of IS_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9126049, upper bound: 123.9123772
time: 11.94 seconds

## Relational analysis of IS_B1_B1_A2

### Relational analysis result of IS_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9061448, upper bound: 123.9067809
time: 15.84 seconds

## BFS IS instance: IS_B1_B2

### Backsubstitution after applying IS history:
0: -69.8902054, 53.0486641, -73.9138641, 55.9102745, -125.8004761, 126.9625244
1: -55.5890274, 49.8915863, -58.5997391, 52.7283821, -108.3174057, 108.4913254
2: -74.8521423, 49.9116287, -79.0853043, 52.6514282, -127.5035629, 128.9969177
3: -81.6211700, 41.7395477, -86.3253326, 44.0380325, -125.6591949, 128.0648804
4: -79.7567215, 52.7206345, -84.6233673, 55.4581528, -135.2148590, 137.3439941
5: -70.5330505, 47.6525917, -74.7420197, 50.0982513, -120.6312943, 122.3946075
6: -73.1721268, 52.1106567, -77.7358246, 54.6438408, -127.8159409, 129.8464813
7: -66.3076172, 60.1448517, -70.0320663, 63.6205254, -129.9281464, 130.1769104
8: -88.5151291, 52.4997902, -93.7299194, 55.0639915, -143.5791168, 146.2297058
9: -61.3551750, 62.5282860, -64.8040314, 66.0537720, -127.4089432, 127.3323135

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=62, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=136, inp2_unstable=142, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=233, inp2_unstable=231, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B1_B2_B1

### Relational analysis result of IS_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9362566, upper bound: 123.9362680
time: 17.38 seconds

## Relational analysis of IS_B1_B2_B2

### Relational analysis result of IS_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9330503, upper bound: 123.9315492
time: 19.55 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -66.8310928, 50.5467186, -74.0325623, 56.3408813, -123.1719742, 124.5792694
1: -52.9668999, 47.7114410, -59.0769806, 52.8421898, -105.8090897, 106.7884216
2: -71.4631729, 47.6612968, -79.4081421, 52.8940964, -124.3572693, 127.0694427
3: -78.0459900, 39.8503151, -86.5053558, 44.2720871, -122.3180771, 126.3556595
4: -76.5393448, 50.1807289, -84.2501984, 56.0987778, -132.6381226, 134.4309082
5: -67.5947571, 45.3055573, -74.5464859, 50.7287827, -118.3235397, 119.8520432
6: -70.4484024, 49.2946320, -77.0402222, 55.7724648, -126.2208710, 126.3348541
7: -63.2310295, 57.5649834, -70.4280624, 63.6546097, -126.8856277, 127.9930344
8: -84.8669510, 49.8649712, -93.5508957, 55.9058456, -140.7727814, 143.4158478
9: -58.5714455, 59.7560463, -65.1169434, 66.2752914, -124.8467178, 124.8729858

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=63, inp2_unstable=63, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=136, inp2_unstable=137, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=238, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9616904, upper bound: 123.9616904
time: 12.58 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9616904, upper bound: 123.9616904
time: 13.97 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -65.1477814, 49.0066299, -70.4546356, 53.4808540, -118.6286316, 119.4612503
1: -51.3121490, 46.4751320, -56.0425797, 50.2907372, -101.6028900, 102.5177078
2: -69.4467621, 46.3225899, -75.4578552, 50.3066025, -119.7533646, 121.7804413
3: -76.0406723, 38.7073631, -82.2868652, 42.0726166, -118.1132889, 120.9942322
4: -74.8772049, 48.5327072, -80.3945618, 53.1528816, -128.0300903, 128.9272461
5: -66.1044006, 43.7729645, -71.0991516, 48.0445976, -114.1489944, 114.8721161
6: -69.2758865, 47.2226257, -73.7449112, 52.5507126, -121.8265991, 120.9675369
7: -61.3677597, 56.1612473, -66.8500061, 60.6275330, -121.9952927, 123.0112534
8: -82.9419785, 48.0569229, -89.2195282, 52.9208908, -135.8628693, 137.2764587
9: -56.9046135, 58.1323357, -61.8548088, 63.0313263, -119.9359360, 119.9871368

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=63, inp2_unstable=63, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=140, inp2_unstable=137, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=216, inp2_unstable=233, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9616904, upper bound: 123.9616904
time: 13.56 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9616904, upper bound: 123.9616904
time: 11.99 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 26.92 seconds
IS_B1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.92
Output dim: 6, lower bound: -123.9126049, upper bound: 123.9123772
IS_B1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.92
Output dim: 6, lower bound: -123.9061448, upper bound: 123.9067809
IS_B1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 26.92
Output dim: 6, lower bound: -123.9362566, upper bound: 123.9362680
IS_B1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 26.92
Output dim: 6, lower bound: -123.9330503, upper bound: 123.9315492
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 26.92
Output dim: 6, lower bound: -123.9616904, upper bound: 123.9616904
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 26.92
Output dim: 6, lower bound: -123.9616904, upper bound: 123.9616904
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 26.92
Output dim: 6, lower bound: -123.9616904, upper bound: 123.9616904
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 26.92
Output dim: 6, lower bound: -123.9616904, upper bound: 123.9616904

## BFS IS instance: IS_B1_B1_A1

### Backsubstitution after applying IS history:
0: -72.5428162, 55.2059097, -75.6133728, 57.4238091, -129.9666138, 130.8192596
1: -57.8811417, 51.7876434, -60.2386093, 53.9670715, -111.8482056, 112.0262451
2: -77.8075943, 51.8546410, -81.0944138, 53.9835625, -131.7911377, 132.9490509
3: -84.7461777, 43.3893852, -88.3493271, 45.1785431, -129.9247131, 131.7386932
4: -82.5672379, 54.9552803, -86.2837524, 57.1157951, -139.6830292, 141.2390137
5: -73.0613327, 49.6962776, -76.2233124, 51.6275253, -124.6888504, 125.9195862
6: -75.5312347, 54.6045074, -78.9345245, 56.7054672, -132.2366943, 133.5390320
7: -68.9963455, 62.3853989, -71.9072495, 65.0277786, -134.0241089, 134.2926483
8: -91.6999741, 54.7920647, -95.6765747, 56.8456573, -148.5456238, 150.4686279
9: -63.7939873, 64.9538651, -66.4788208, 67.6743927, -131.4683838, 131.4326477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=63, inp2_unstable=62, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=135, inp2_unstable=138, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=234, inp2_unstable=238, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_B1_B1_A1_B1

### Relational analysis result of IS_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9061448, upper bound: 123.9067809
time: 18.52 seconds

## Relational analysis of IS_B1_B1_A1_B2

### Relational analysis result of IS_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9061448, upper bound: 123.9067809
time: 16.52 seconds

## BFS IS instance: IS_B1_B1_A2

### Backsubstitution after applying IS history:
0: -82.0353394, 62.4043884, -74.8209534, 56.8187370, -138.8540497, 137.2253418
1: -65.5461807, 58.4676857, -59.6016731, 53.4050598, -118.9512177, 118.0693588
2: -87.9929657, 58.6037483, -80.2400742, 53.4315033, -141.4244385, 138.8438263
3: -95.9253082, 49.0702858, -87.4139709, 44.7095261, -140.6347961, 136.4842529
4: -93.1607819, 62.2472343, -85.3863220, 56.5050049, -149.6657867, 147.6335602
5: -82.4728851, 56.3010712, -75.4357910, 51.0760880, -133.5489655, 131.7368622
6: -85.0128326, 62.0688400, -78.1345978, 56.0793953, -141.0922089, 140.2034302
7: -78.1521149, 70.5219803, -71.1451187, 64.3532104, -142.5053101, 141.6670837
8: -103.4968033, 61.9284859, -94.6942139, 56.2502289, -159.7470245, 156.6226959
9: -72.1841736, 73.3554306, -65.7711258, 66.9712601, -139.1554260, 139.1265564

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=63, inp2_unstable=62, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=142, inp2_unstable=138, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=243, inp2_unstable=237, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_B1_B1_A2_A1

### Relational analysis result of IS_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9061448, upper bound: 123.9067809
time: 17.46 seconds

## Relational analysis of IS_B1_B1_A2_A2

### Relational analysis result of IS_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9061448, upper bound: 123.9067809
time: 14.27 seconds

## BFS IS instance: IS_B1_B2_B1

### Backsubstitution after applying IS history:
0: -67.2865067, 50.9437637, -66.7120285, 50.0855637, -117.3720703, 117.6557846
1: -53.3810997, 48.0370026, -52.5227890, 47.5847740, -100.9658661, 100.5597916
2: -71.9732437, 48.0225372, -71.1291351, 47.4304008, -119.4036407, 119.1516571
3: -78.5514297, 40.1445236, -77.8552170, 39.6235580, -118.1749802, 117.9997330
4: -76.9518661, 50.5662155, -76.8506165, 49.5102730, -126.4621124, 127.4168320
5: -68.0139084, 45.6797485, -67.7800674, 44.6478157, -112.6617203, 113.4598160
6: -70.7823486, 49.7456360, -71.0936661, 48.1199608, -118.9023132, 120.8392944
7: -63.7117004, 57.9398956, -62.8679581, 57.5214272, -121.2331238, 120.8078537
8: -85.3740234, 50.3055267, -85.0358582, 48.9924431, -134.3664703, 135.3413849
9: -58.9716797, 60.1567993, -58.2274361, 59.5071564, -118.4788284, 118.3842316

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=61, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=136, inp2_unstable=141, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=226, inp2_unstable=217, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 181

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_B1_B2_B1_A1

### Relational analysis result of IS_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9362566, upper bound: 123.9362680
time: 14.89 seconds

## Relational analysis of IS_B1_B2_B1_A2

### Relational analysis result of IS_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9362566, upper bound: 123.9362680
time: 14.82 seconds

## BFS IS instance: IS_B1_B2_B2

### Backsubstitution after applying IS history:
0: -65.0811920, 49.1242676, -65.0264740, 48.4377747, -113.5189667, 114.1507416
1: -51.4783783, 46.4610519, -50.8121567, 46.3392410, -97.8176193, 97.2732086
2: -69.5113144, 46.4093132, -69.0719681, 46.0534210, -115.5647354, 115.4812775
3: -75.9722290, 38.7808723, -75.9365463, 38.4915695, -114.4637909, 114.7173996
4: -74.6089706, 48.7118759, -75.3305283, 47.7851486, -122.3941193, 124.0424042
5: -65.8917694, 43.9699135, -66.3196793, 43.0095215, -108.9012909, 110.2895966
6: -68.8225327, 47.6702080, -70.1064529, 45.8771477, -114.6996765, 117.7766571
7: -61.4894981, 56.0805359, -60.9561996, 56.1447678, -117.6342621, 117.0367279
8: -82.7351685, 48.3752747, -83.2056427, 47.0072861, -129.7424622, 131.5809174
9: -56.9450302, 58.1391754, -56.5486374, 57.8686485, -114.8136749, 114.6877899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=61, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=135, inp2_unstable=143, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=218, inp2_unstable=206, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B1_B2_B2_A1

### Relational analysis result of IS_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9330503, upper bound: 123.9315492
time: 20.99 seconds

## Relational analysis of IS_B1_B2_B2_A2

### Relational analysis result of IS_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9330503, upper bound: 123.9315492
time: 17.65 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -66.8310928, 50.5467186, -66.6451340, 50.4037933, -117.2348862, 117.1918335
1: -52.9668999, 47.7114410, -52.8169708, 47.5798035, -100.5467072, 100.5284042
2: -71.4631729, 47.6612968, -71.2630234, 47.5310974, -118.9942627, 118.9243164
3: -78.0459900, 39.8503151, -77.8260803, 39.7404060, -117.7863922, 117.6763916
4: -76.5393448, 50.1807289, -76.3287811, 50.0378799, -126.5772247, 126.5095062
5: -67.5947571, 45.3055573, -67.4079895, 45.1763649, -112.7711105, 112.7135468
6: -70.4484024, 49.2946320, -70.2597351, 49.1492729, -119.5976715, 119.5543442
7: -63.2310295, 57.5649834, -63.0522804, 57.4057808, -120.6368103, 120.6172485
8: -84.8669510, 49.8649712, -84.6345367, 49.7255096, -134.5924377, 134.4995117
9: -58.5714455, 59.7560463, -58.4069595, 59.5901184, -118.1615601, 118.1630096

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=63, inp2_unstable=62, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=136, inp2_unstable=136, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_B2_A1_B1_B1

### Relational analysis result of IS_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9008917, upper bound: 123.9025544
time: 15.99 seconds

## Relational analysis of IS_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B1_B1

### Relational analysis result of IS_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.8992163, upper bound: 123.9011364
time: 13.74 seconds

## Relational analysis of IS_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9548171, upper bound: 123.9533852
time: 15.38 seconds

## Relational analysis of IS_B2_A1_B1_A2

### Relational analysis result of IS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9531329, upper bound: 123.9524301
time: 16.88 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -66.8310928, 50.5467186, -64.9603806, 48.8624153, -115.6935043, 115.5070953
1: -52.9668999, 47.7114410, -51.1609573, 46.3422127, -99.3091125, 98.8723984
2: -71.4631729, 47.6612968, -69.2449417, 46.1911850, -117.6543579, 116.9062347
3: -78.0459900, 39.8503151, -75.8194962, 38.5966873, -116.6426697, 115.6697998
4: -76.5393448, 50.1807289, -74.6646500, 48.3888702, -124.9282150, 124.8453827
5: -67.5947571, 45.3055573, -65.9160614, 43.6428146, -111.2375717, 111.2216187
6: -70.4484024, 49.2946320, -69.0851059, 47.0767746, -117.5251694, 118.3797302
7: -63.2310295, 57.5649834, -61.1875916, 56.0004082, -119.2314224, 118.7525711
8: -84.8669510, 49.8649712, -82.7078171, 47.9162827, -132.7832184, 132.5727539
9: -58.5714455, 59.7560463, -56.7387314, 57.9648590, -116.5362930, 116.4947815

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=63, inp2_unstable=62, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=136, inp2_unstable=140, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=216, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9561518, upper bound: 123.9551151
time: 15.15 seconds

## Relational analysis of IS_B2_A1_B2_A2

### Relational analysis result of IS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9558614, upper bound: 123.9549752
time: 14.87 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -65.1477814, 49.0066299, -66.6451340, 50.4037933, -115.5515747, 115.6517410
1: -51.3121490, 46.4751320, -52.8169708, 47.5798035, -98.8919525, 99.2920914
2: -69.4467621, 46.3225899, -71.2630234, 47.5310974, -116.9778442, 117.5856171
3: -76.0406723, 38.7073631, -77.8260803, 39.7404060, -115.7810822, 116.5334473
4: -74.8772049, 48.5327072, -76.3287811, 50.0378799, -124.9150696, 124.8614883
5: -66.1044006, 43.7729645, -67.4079895, 45.1763649, -111.2807541, 111.1809387
6: -69.2758865, 47.2226257, -70.2597351, 49.1492729, -118.4251556, 117.4823532
7: -61.3677597, 56.1612473, -63.0522804, 57.4057808, -118.7735443, 119.2135162
8: -82.9419785, 48.0569229, -84.6345367, 49.7255096, -132.6674805, 132.6914673
9: -56.9046135, 58.1323357, -58.4069595, 59.5901184, -116.4947357, 116.5392838

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=63, inp2_unstable=62, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=140, inp2_unstable=136, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=216, inp2_unstable=224, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B2_A2_B1_B1

### Relational analysis result of IS_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9547309, upper bound: 123.9551878
time: 14.07 seconds

## Relational analysis of IS_B2_A2_B1_B2

### Relational analysis result of IS_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9543876, upper bound: 123.9543876
time: 13.58 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -65.1477814, 49.0066299, -64.9603806, 48.8624153, -114.0101929, 113.9669952
1: -51.3121490, 46.4751320, -51.1609573, 46.3422127, -97.6543579, 97.6360855
2: -69.4467621, 46.3225899, -69.2449417, 46.1911850, -115.6379471, 115.5675354
3: -76.0406723, 38.7073631, -75.8194962, 38.5966873, -114.6373444, 114.5268555
4: -74.8772049, 48.5327072, -74.6646500, 48.3888702, -123.2660599, 123.1973572
5: -66.1044006, 43.7729645, -65.9160614, 43.6428146, -109.7472153, 109.6890182
6: -69.2758865, 47.2226257, -69.0851059, 47.0767746, -116.3526535, 116.3077316
7: -61.3677597, 56.1612473, -61.1875916, 56.0004082, -117.3681641, 117.3488388
8: -82.9419785, 48.0569229, -82.7078171, 47.9162827, -130.8582611, 130.7647247
9: -56.9046135, 58.1323357, -56.7387314, 57.9648590, -114.8694611, 114.8710556

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=63, inp2_unstable=62, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=140, inp2_unstable=140, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=216, inp2_unstable=216, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B2_A2_B2_A1

### Relational analysis result of IS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9551878, upper bound: 123.9547309
time: 15.84 seconds

## Relational analysis of IS_B2_A2_B2_A2

### Relational analysis result of IS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9543876, upper bound: 123.9543876
time: 13.37 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 34.82 seconds
IS_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 34.82
Output dim: 6, lower bound: -123.9061448, upper bound: 123.9067809
IS_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 34.82
Output dim: 6, lower bound: -123.9061448, upper bound: 123.9067809
IS_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 34.82
Output dim: 6, lower bound: -123.9061448, upper bound: 123.9067809
IS_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 34.82
Output dim: 6, lower bound: -123.9061448, upper bound: 123.9067809
IS_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 34.82
Output dim: 6, lower bound: -123.9362566, upper bound: 123.9362680
IS_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 34.82
Output dim: 6, lower bound: -123.9362566, upper bound: 123.9362680
IS_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 34.82
Output dim: 6, lower bound: -123.9330503, upper bound: 123.9315492
IS_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 34.82
Output dim: 6, lower bound: -123.9330503, upper bound: 123.9315492
IS_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 34.82
Output dim: 6, lower bound: -123.9548171, upper bound: 123.9533852
IS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 34.82
Output dim: 6, lower bound: -123.9531329, upper bound: 123.9524301
IS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 34.82
Output dim: 6, lower bound: -123.9561518, upper bound: 123.9551151
IS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 34.82
Output dim: 6, lower bound: -123.9558614, upper bound: 123.9549752
IS_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 34.82
Output dim: 6, lower bound: -123.9547309, upper bound: 123.9551878
IS_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 34.82
Output dim: 6, lower bound: -123.9543876, upper bound: 123.9543876
IS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 34.82
Output dim: 6, lower bound: -123.9551878, upper bound: 123.9547309
IS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 34.82
Output dim: 6, lower bound: -123.9543876, upper bound: 123.9543876

## BFS IS instance: IS_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -72.5428162, 55.2059097, -74.6845245, 56.7161980, -129.2590179, 129.8904114
1: -57.8811417, 51.7876434, -59.4930038, 53.3097725, -111.1909103, 111.2806473
2: -77.8075943, 51.8546410, -80.0943604, 53.3365631, -131.1441650, 131.9490051
3: -84.7461777, 43.3893852, -87.2517319, 44.6278458, -129.3740234, 130.6411133
4: -82.5672379, 54.9552803, -85.2347031, 56.4009514, -138.9681854, 140.1899719
5: -73.0613327, 49.6962776, -75.3006592, 50.9831810, -124.0445023, 124.9969177
6: -75.5312347, 54.6045074, -77.9951096, 55.9735489, -131.5047913, 132.5996094
7: -68.9963455, 62.3853989, -71.0133667, 64.2377853, -133.2341309, 133.3987579
8: -91.6999741, 54.7920647, -94.5252686, 56.1479416, -147.8479156, 149.3173065
9: -63.7939873, 64.9538651, -65.6506424, 66.8514862, -130.6454773, 130.6044922

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=63, inp2_unstable=61, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=135, inp2_unstable=138, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=234, inp2_unstable=236, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 42

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_B1_B1_A1_B1_B1

### Relational analysis result of IS_B1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.8991338, upper bound: 123.8997756
time: 19.10 seconds

## Relational analysis of IS_B1_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 17.10 + 585.30 = 602.39 seconds
