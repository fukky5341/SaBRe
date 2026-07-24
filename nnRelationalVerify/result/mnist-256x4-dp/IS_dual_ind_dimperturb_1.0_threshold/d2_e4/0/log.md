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
execution time: IAR + RelationalAnalysis = 1.32 + 15.79 = 17.11 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -123.9695598, upper bound: 123.9695598

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 139

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9646370, upper bound: 123.9643598
time: 14.57 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9642107, upper bound: 123.9642107
time: 14.24 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 28.93 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 28.93
Output dim: 6, lower bound: -123.9646370, upper bound: 123.9643598
IS_A2, status: Status.UNKNOWN, split count: 1, time: 28.93
Output dim: 6, lower bound: -123.9642107, upper bound: 123.9642107

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -69.4722824, 52.6508408, -74.8761520, 56.9937286, -126.4659882, 127.5269699
1: -55.2117119, 49.5868416, -59.7785492, 53.4377823, -108.6494904, 109.3653870
2: -74.3503723, 49.5872612, -80.3173141, 53.5004196, -127.8507919, 129.9045715
3: -81.1151886, 41.4849548, -87.4799042, 44.7900848, -125.9052734, 128.9648590
4: -79.3388062, 52.2857208, -85.1500626, 56.7556725, -136.0944824, 137.4357758
5: -70.1595383, 47.2455902, -75.3637848, 51.3312645, -121.4907990, 122.6093750
6: -72.8572769, 51.5764389, -77.8253098, 56.4721069, -129.3293762, 129.4017487
7: -65.8759766, 59.7949753, -71.2615967, 64.3645706, -130.2405396, 131.0565643
8: -88.0475845, 52.0366173, -94.5679779, 56.5714073, -144.6189880, 146.6045837
9: -60.9206047, 62.1057129, -65.8574142, 67.0164719, -127.9370575, 127.9631042

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=63, inp2_unstable=64, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=138, inp2_unstable=138, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=239, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 139

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9556549, upper bound: 123.9545292
time: 16.08 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9614492, upper bound: 123.9612166
time: 17.83 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -68.5348892, 51.6483727, -72.5886765, 55.1339722, -123.6688385, 124.2370453
1: -54.1597176, 48.8899994, -57.8122101, 51.8036957, -105.9634094, 106.7022095
2: -73.1413498, 48.7803650, -77.7681732, 51.8341217, -124.9754715, 126.5485229
3: -80.0822144, 40.8083000, -84.8071976, 43.3792686, -123.4614868, 125.6154861
4: -78.6140900, 51.2071953, -82.7314301, 54.8327141, -133.4468079, 133.9386292
5: -69.4163666, 46.2085648, -73.1811295, 49.5704918, -118.9868622, 119.3896790
6: -72.5381851, 50.0702324, -75.7927094, 54.3308372, -126.8690186, 125.8629456
7: -64.7247849, 59.0564041, -68.9549408, 62.4416046, -127.1663895, 128.0113525
8: -87.1227951, 50.7388000, -91.8303452, 54.5961189, -141.7189026, 142.5691528
9: -59.9385452, 61.1711502, -63.7609024, 64.9317856, -124.8702850, 124.9320526

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=63, inp2_unstable=64, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=138, inp2_unstable=138, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=222, inp2_unstable=234, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 139

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9553527, upper bound: 123.9544204
time: 17.87 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9610872, upper bound: 123.9610872
time: 15.79 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 35.02 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 35.02
Output dim: 6, lower bound: -123.9556549, upper bound: 123.9545292
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 35.02
Output dim: 6, lower bound: -123.9614492, upper bound: 123.9612166
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 35.02
Output dim: 6, lower bound: -123.9553527, upper bound: 123.9544204
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 35.02
Output dim: 6, lower bound: -123.9610872, upper bound: 123.9610872

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -62.5534286, 47.0202141, -58.4875336, 43.6826401, -106.2360687, 105.5077438
1: -49.2730484, 44.6175919, -45.7037888, 41.6678848, -90.9409332, 90.3213730
2: -66.6421204, 44.5365105, -62.0561790, 41.5317841, -108.1739044, 106.5926743
3: -72.9652863, 37.1571045, -68.1648026, 34.5691872, -107.5344543, 105.3219070
4: -71.8256302, 46.5331345, -67.3960495, 43.1248398, -114.9504471, 113.9291840
5: -63.4719429, 42.0185623, -59.5620422, 38.9258766, -102.3978195, 101.5805969
6: -66.4775696, 45.2274590, -62.7523346, 41.4252129, -107.9027863, 107.9797821
7: -58.9144440, 53.9263344, -54.7683029, 50.4784317, -109.3928757, 108.6946411
8: -79.6022415, 46.0917091, -74.5684357, 42.5011826, -122.1034164, 120.6601410
9: -54.5595016, 55.7552223, -50.8127670, 52.0033798, -106.5628815, 106.5679779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=63, inp2_unstable=63, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=138, inp2_unstable=139, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=214, inp2_unstable=205, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 85

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9504628, upper bound: 123.9494092
time: 14.62 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9524486, upper bound: 123.9513490
time: 17.32 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -66.9136963, 50.5760880, -66.3723831, 50.1175842, -117.0312805, 116.9484711
1: -53.0136948, 47.7587585, -52.4812012, 47.3535423, -100.3672333, 100.2399521
2: -71.5092239, 47.7229919, -70.8631363, 47.3003197, -118.8095398, 118.5861282
3: -78.1149063, 39.8876915, -77.4825821, 39.4991989, -117.6141052, 117.3702698
4: -76.5811462, 50.1608353, -75.9926834, 49.6874313, -126.2685776, 126.1535187
5: -67.6967773, 45.3138924, -67.1903763, 44.9071884, -112.6039658, 112.5042725
6: -70.5192642, 49.2290878, -70.0472260, 48.6714325, -119.1906967, 119.2763138
7: -63.3017464, 57.6315041, -62.7020416, 57.1757126, -120.4774475, 120.3335419
8: -84.9369049, 49.8489761, -84.2089920, 49.3114471, -134.2483521, 134.0579681
9: -58.5749664, 59.7706566, -58.0707626, 59.2488556, -117.8238144, 117.8414154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=63, inp2_unstable=63, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=138, inp2_unstable=138, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=221, inp2_unstable=222, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9568805, upper bound: 123.9566513
time: 16.39 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9580803, upper bound: 123.9578957
time: 16.43 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -61.8593903, 46.1973801, -56.6085701, 42.1257324, -103.9851227, 102.8059387
1: -48.4194374, 44.0848389, -44.0791779, 40.3163567, -88.7357864, 88.1640091
2: -65.6936340, 43.8896637, -59.9637184, 40.1507187, -105.8443527, 103.8533707
3: -72.2220459, 36.6401863, -65.9760895, 33.4184341, -105.6404648, 102.6162720
4: -71.3606186, 45.6491394, -65.4048767, 41.5495491, -112.9101639, 111.0540161
5: -62.9680748, 41.1387100, -57.7616959, 37.4541550, -100.4222260, 98.9004059
6: -66.3852844, 43.9271240, -61.0906181, 39.6620674, -106.0473480, 105.0177307
7: -58.0108070, 53.3896980, -52.8902931, 48.8928604, -106.9036560, 106.2799835
8: -78.9604721, 44.9697151, -72.3221512, 40.8416176, -119.8020935, 117.2918472
9: -53.8078156, 55.0426445, -49.0997963, 50.2882690, -104.0960693, 104.1424332

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=63, inp2_unstable=63, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=137, inp2_unstable=138, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=210, inp2_unstable=202, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9500813, upper bound: 123.9492449
time: 18.55 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9521612, upper bound: 123.9512429
time: 17.69 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -66.1774368, 49.7239227, -64.4074554, 48.4950485, -114.6724854, 114.1313553
1: -52.1305656, 47.1991272, -50.7867012, 45.9467049, -98.0772705, 97.9858246
2: -70.5141449, 47.0550385, -68.6644440, 45.8639145, -116.3780594, 115.7194672
3: -77.3156052, 39.3316498, -75.1902542, 38.2838554, -115.5994568, 114.5218964
4: -76.0643921, 49.2422523, -73.9042511, 48.0325470, -124.0969315, 123.1464996
5: -67.1440048, 44.4157219, -65.2985687, 43.3813057, -110.5253143, 109.7142944
6: -70.3832321, 47.8948517, -68.3066864, 46.8183746, -117.2016068, 116.2015381
7: -62.3516388, 57.0583801, -60.7202911, 55.5181923, -117.8698196, 117.7786636
8: -84.2434464, 48.7009659, -81.8562012, 47.5867424, -131.8301849, 130.5571594
9: -57.7753792, 59.0085106, -56.2653847, 57.4486618, -115.2240448, 115.2738876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=63, inp2_unstable=63, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=137, inp2_unstable=138, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=216, inp2_unstable=216, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9564749, upper bound: 123.9564525
time: 15.23 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9577807, upper bound: 123.9577807
time: 14.50 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 31.08 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 31.08
Output dim: 6, lower bound: -123.9504628, upper bound: 123.9494092
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 31.08
Output dim: 6, lower bound: -123.9524486, upper bound: 123.9513490
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 31.08
Output dim: 6, lower bound: -123.9568805, upper bound: 123.9566513
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 31.08
Output dim: 6, lower bound: -123.9580803, upper bound: 123.9578957
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 31.08
Output dim: 6, lower bound: -123.9500813, upper bound: 123.9492449
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 31.08
Output dim: 6, lower bound: -123.9521612, upper bound: 123.9512429
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 31.08
Output dim: 6, lower bound: -123.9564749, upper bound: 123.9564525
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 31.08
Output dim: 6, lower bound: -123.9577807, upper bound: 123.9577807

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -52.7326965, 38.9792595, -54.3623428, 40.2990608, -93.0317535, 93.3415909
1: -40.8032074, 37.5553360, -42.1437531, 38.7013435, -79.5045471, 79.6990891
2: -55.7191353, 37.2997398, -57.4750404, 38.4922562, -94.2113953, 94.7747803
3: -61.4755249, 31.1120129, -63.3362122, 32.0336571, -93.5091782, 94.4482117
4: -61.2613258, 38.3812141, -62.9616852, 39.6999969, -100.9613190, 101.3428955
5: -54.0462112, 34.5137863, -55.6062088, 35.7623787, -89.8085938, 90.1199875
6: -57.5220222, 36.2156906, -58.9991989, 37.6420631, -95.1640854, 95.2148895
7: -49.0828056, 45.5984077, -50.6423569, 46.9802208, -96.0630188, 96.2407684
8: -67.6492157, 37.5535774, -69.5510712, 38.9160767, -106.5652924, 107.1046448
9: -45.6233635, 46.7716370, -47.0615196, 48.2305145, -93.8538818, 93.8331528

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=63, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=138, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=186, inp2_unstable=197, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 139

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9504628, upper bound: 123.9494092
time: 17.18 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9504628, upper bound: 123.9494092
time: 16.55 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -57.0419540, 42.5106392, -56.6326942, 42.1635933, -99.2055511, 99.1433334
1: -44.5219612, 40.6604233, -44.1032829, 40.3340530, -84.8560181, 84.7637024
2: -60.5047607, 40.4830818, -59.9948769, 40.1670723, -100.6718292, 100.4779587
3: -66.5267639, 33.7659340, -65.9946518, 33.4285851, -99.9553528, 99.7605896
4: -65.8828506, 41.9680862, -65.3982544, 41.5882950, -107.4711456, 107.3663406
5: -58.1711426, 37.8125610, -57.7797661, 37.5062370, -95.6773834, 95.5923309
6: -61.4468231, 40.1814919, -61.0627861, 39.7293625, -101.1761780, 101.2442703
7: -53.3863449, 49.2514114, -52.9123611, 48.9050560, -102.2913818, 102.1637726
8: -72.8986130, 41.3287010, -72.3141632, 40.8966217, -113.7952347, 113.6428604
9: -49.5313911, 50.7278137, -49.1239090, 50.3094254, -99.8407974, 99.8517227

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=63, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=136, inp2_unstable=138, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=205, inp2_unstable=202, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 139

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9524484, upper bound: 123.9513490
time: 19.00 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9524484, upper bound: 123.9513490
time: 16.05 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -56.6741219, 42.2039375, -61.9994621, 46.5472908, -103.2214127, 104.2033920
1: -44.1971283, 40.4098091, -48.7183762, 44.2203255, -88.4174271, 89.1281815
2: -60.1169357, 40.2022896, -65.9964218, 44.0986366, -104.2155609, 106.1987152
3: -66.1496048, 33.5620193, -72.3631134, 36.7949219, -102.9445190, 105.9251251
4: -65.5647812, 41.6496887, -71.2865677, 46.0451698, -111.6099472, 112.9362564
5: -57.8612289, 37.5128250, -62.9875755, 41.5763245, -99.4375381, 100.5003891
6: -61.1840858, 39.8250771, -66.0673828, 44.6551666, -105.8392410, 105.8924561
7: -53.0430222, 48.9561768, -58.3143539, 53.4738884, -106.5169067, 107.2705307
8: -72.4830093, 40.9770470, -78.8877258, 45.5352936, -118.0183029, 119.8647766
9: -49.2339401, 50.4034462, -54.0761261, 55.2468224, -104.4807587, 104.4795685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=63, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=134, inp2_unstable=137, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=206, inp2_unstable=213, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9568805, upper bound: 123.9566513
time: 13.99 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9568805, upper bound: 123.9494092
time: 13.73 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -61.0589676, 45.7924728, -64.2521744, 48.3850365, -109.4439926, 110.0446472
1: -47.9698029, 43.5588455, -50.6555901, 45.8339844, -93.8037796, 94.2144089
2: -64.9887238, 43.4253769, -68.5007629, 45.7474365, -110.7361526, 111.9261398
3: -71.2646484, 36.2609787, -74.9947281, 38.1858711, -109.4505157, 111.2556915
4: -70.2671280, 45.2895927, -73.7061996, 47.9218292, -118.1889267, 118.9957886
5: -62.0634689, 40.8532906, -65.1466599, 43.2929077, -105.3563690, 105.9999390
6: -65.1742630, 43.8516197, -68.1136246, 46.7226791, -111.8969345, 111.9652328
7: -57.4176102, 52.6670227, -60.5711288, 55.3786507, -112.7962646, 113.2381287
8: -77.8163376, 44.7918549, -81.6302795, 47.4823112, -125.2986450, 126.4221268
9: -53.2150345, 54.4094849, -56.1294289, 57.3046036, -110.5196152, 110.5389099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=63, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=136, inp2_unstable=138, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=213, inp2_unstable=216, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9580803, upper bound: 123.9578957
time: 12.65 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9580803, upper bound: 123.9578957
time: 18.57 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -52.8240013, 38.8053169, -52.6373901, 38.8641815, -91.6881866, 91.4427032
1: -40.6036949, 37.5737267, -40.6485062, 37.4551926, -78.0588760, 78.2222290
2: -55.6320763, 37.2120247, -55.5486450, 37.2122040, -92.8442841, 92.7606659
3: -61.6075516, 31.0620117, -61.3258820, 30.9789009, -92.5864487, 92.3878937
4: -61.6289368, 38.1416016, -61.1251717, 38.2468948, -99.8758316, 99.2667694
5: -54.3148689, 34.2324638, -53.9517593, 34.4064522, -88.7213135, 88.1842194
6: -58.1308289, 35.6075783, -57.4634781, 36.0166397, -94.1474609, 93.0710373
7: -48.9524155, 45.7274323, -48.9088440, 45.5174980, -94.4699097, 94.6362610
8: -67.9223175, 37.0842552, -67.4837723, 37.3801994, -105.3025208, 104.5680237
9: -45.6085663, 46.7374115, -45.4914398, 46.6457977, -92.2543640, 92.2288513

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=63, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=136, inp2_unstable=138, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=170, inp2_unstable=189, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 139

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9497413, upper bound: 123.9489758
time: 17.49 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9500813, upper bound: 123.9492449
time: 13.40 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -56.6133575, 41.8962669, -54.8020439, 40.6430092, -97.2563629, 96.6983109
1: -43.8760719, 40.3062630, -42.5188560, 39.0162659, -82.8923340, 82.8251190
2: -59.8464622, 40.0184708, -57.9554214, 38.8181496, -98.6646118, 97.9738922
3: -66.0710907, 33.4081039, -63.8620987, 32.3106384, -98.3817291, 97.2701950
4: -65.7068787, 41.2903519, -63.4565620, 40.0528259, -105.7597046, 104.7469177
5: -57.9315681, 37.1243591, -56.0240631, 36.0695496, -94.0011139, 93.1484222
6: -61.5995903, 39.0926170, -59.4395447, 38.0086021, -99.6081696, 98.5321655
7: -52.7460442, 48.9336739, -51.0801468, 47.3587189, -100.1047440, 100.0138245
8: -72.5698929, 40.4075966, -70.1266174, 39.2747841, -111.8446808, 110.5342102
9: -49.0229034, 50.2373466, -47.4544373, 48.6347427, -97.6576462, 97.6917725

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=63, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=136, inp2_unstable=138, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=198, inp2_unstable=198, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 139

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9518894, upper bound: 123.9511223
time: 14.97 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9518891, upper bound: 123.9512428
time: 16.97 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -56.4856796, 41.7911415, -60.1500511, 45.0173187, -101.5029984, 101.9411926
1: -43.7661057, 40.2285271, -47.1213760, 42.8918991, -86.6579971, 87.3498840
2: -59.7233467, 39.9088860, -63.9285736, 42.7366104, -102.4599609, 103.8374405
3: -65.9695663, 33.3553658, -70.2130203, 35.6565399, -101.6261063, 103.5683899
4: -65.6431274, 41.1820107, -69.3232574, 44.4888916, -110.1320190, 110.5052567
5: -57.8489876, 37.0203247, -61.2110329, 40.1354599, -97.9844513, 98.2313309
6: -61.5424080, 38.9632492, -64.4279480, 42.9101486, -104.4525604, 103.3911972
7: -52.6413765, 48.8430748, -56.4516563, 51.9117355, -104.5531158, 105.2947311
8: -72.4437256, 40.2723694, -76.6757278, 43.9035301, -116.3472519, 116.9480743
9: -48.9413147, 50.1364899, -52.3804817, 53.5570869, -102.4983978, 102.5169601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=63, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=136, inp2_unstable=137, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=192, inp2_unstable=211, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9558915, upper bound: 123.9559217
time: 27.11 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9564749, upper bound: 123.9564525
time: 11.81 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -60.5369186, 45.1072922, -62.3418121, 46.8058624, -107.3427734, 107.4490967
1: -47.2622375, 43.1481171, -49.0063286, 44.4644051, -91.7266235, 92.1544495
2: -64.2288437, 42.9022446, -66.3622742, 44.3455009, -108.5743408, 109.2645187
3: -70.7173615, 35.8482323, -72.7709808, 37.0062752, -107.7236252, 108.6192093
4: -69.9907837, 44.5448036, -71.6761551, 46.3108940, -116.3016815, 116.2209625
5: -61.7239876, 40.1063423, -63.3110657, 41.8050117, -103.5289993, 103.4173889
6: -65.2414246, 42.7002182, -66.4204254, 44.9196968, -110.1611176, 109.1206284
7: -56.6868324, 52.2759438, -58.6437569, 53.7652855, -110.4521179, 110.9197006
8: -77.3847656, 43.8093300, -79.3423386, 45.8000717, -123.1848373, 123.1516724
9: -52.6216621, 53.8465271, -54.3743439, 55.5552139, -108.1768799, 108.2208710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=63, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=136, inp2_unstable=137, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=209, inp2_unstable=213, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9571480, upper bound: 123.9571984
time: 13.80 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9577807, upper bound: 123.9577807
time: 11.28 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 26.48 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.48
Output dim: 6, lower bound: -123.9504628, upper bound: 123.9494092
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.48
Output dim: 6, lower bound: -123.9504628, upper bound: 123.9494092
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.48
Output dim: 6, lower bound: -123.9524484, upper bound: 123.9513490
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.48
Output dim: 6, lower bound: -123.9524484, upper bound: 123.9513490
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.48
Output dim: 6, lower bound: -123.9568805, upper bound: 123.9566513
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.48
Output dim: 6, lower bound: -123.9568805, upper bound: 123.9494092
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.48
Output dim: 6, lower bound: -123.9580803, upper bound: 123.9578957
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.48
Output dim: 6, lower bound: -123.9580803, upper bound: 123.9578957
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.48
Output dim: 6, lower bound: -123.9497413, upper bound: 123.9489758
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.48
Output dim: 6, lower bound: -123.9500813, upper bound: 123.9492449
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.48
Output dim: 6, lower bound: -123.9518894, upper bound: 123.9511223
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.48
Output dim: 6, lower bound: -123.9518891, upper bound: 123.9512428
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.48
Output dim: 6, lower bound: -123.9558915, upper bound: 123.9559217
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.48
Output dim: 6, lower bound: -123.9564749, upper bound: 123.9564525
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.48
Output dim: 6, lower bound: -123.9571480, upper bound: 123.9571984
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.48
Output dim: 6, lower bound: -123.9577807, upper bound: 123.9577807

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -52.7326965, 38.9792595, -50.4485779, 37.1068382, -89.8395386, 89.4278412
1: -40.8032074, 37.5553360, -38.8212662, 35.8860817, -76.6892853, 76.3766022
2: -55.7191353, 37.2997398, -53.1431999, 35.6249161, -91.3440552, 90.4429398
3: -61.4755249, 31.1120129, -58.7292786, 29.6574459, -91.1329727, 89.8412704
4: -61.2613258, 38.3812141, -58.6873398, 36.4796066, -97.7409286, 97.0685577
5: -54.0462112, 34.5137863, -51.8143768, 32.7688065, -86.8150177, 86.3281631
6: -57.5220222, 36.2156906, -55.3521233, 34.1278648, -91.6498871, 91.5678101
7: -49.0828056, 45.5984077, -46.7457581, 43.6465721, -92.7293777, 92.3441544
8: -67.6492157, 37.5535774, -64.7907791, 35.5732040, -103.2224121, 102.3443375
9: -45.6233635, 46.7716370, -43.5142059, 44.6426926, -90.2660522, 90.2858353

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=62, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=137, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=186, inp2_unstable=176, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 139

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9502679, upper bound: 123.9491371
time: 14.30 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9504628, upper bound: 123.9494092
time: 17.92 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -52.7326965, 38.9792595, -50.5695343, 36.9559021, -89.6885986, 89.5487823
1: -40.8032074, 37.5553360, -38.6372375, 35.9264603, -76.7296677, 76.1925735
2: -55.7191353, 37.2997398, -53.0859108, 35.5557327, -91.2748642, 90.3856506
3: -61.4755249, 31.1120129, -58.8999863, 29.6163788, -91.0919037, 90.0119934
4: -61.2613258, 38.3812141, -59.0947762, 36.2528915, -97.5142136, 97.4759903
5: -54.0462112, 34.5137863, -52.1193619, 32.5032806, -86.5494919, 86.6331482
6: -57.5220222, 36.2156906, -55.9978485, 33.5292969, -91.0513077, 92.2135391
7: -49.0828056, 45.5984077, -46.6448517, 43.8025017, -92.8853073, 92.2432556
8: -67.6492157, 37.5535774, -65.0895233, 35.1101112, -102.7593231, 102.6430969
9: -45.6233635, 46.7716370, -43.5288200, 44.6257896, -90.2491531, 90.3004608

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=62, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=137, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=186, inp2_unstable=165, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 139

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9502679, upper bound: 123.9491371
time: 15.30 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9504628, upper bound: 123.9494092
time: 16.44 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -57.0419540, 42.5106392, -52.4662704, 38.7633858, -95.8053436, 94.9769135
1: -44.5219612, 40.6604233, -40.5705376, 37.3435440, -81.8655090, 81.2309570
2: -60.5047607, 40.4830818, -55.3916931, 37.1246109, -97.6293716, 95.8747711
3: -66.5267639, 33.7659340, -61.1057358, 30.9061337, -97.4328995, 94.8716583
4: -65.8828506, 41.9680862, -60.8721924, 38.1563492, -104.0391998, 102.8402710
5: -58.1711426, 37.8125610, -53.7486229, 34.3210907, -92.4922333, 91.5611877
6: -61.4468231, 40.1814919, -57.1979828, 35.9817657, -97.4285889, 97.3794708
7: -53.3863449, 49.2514114, -48.7774544, 45.3591843, -98.7454987, 98.0288696
8: -72.8986130, 41.3287010, -67.2755127, 37.3434372, -110.2420502, 108.6042099
9: -49.5313911, 50.7278137, -45.3296928, 46.5097809, -96.0411682, 96.0575104

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=62, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=136, inp2_unstable=137, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=205, inp2_unstable=187, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 139

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9522437, upper bound: 123.9511712
time: 16.72 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -123.9524486, upper bound: 123.9513490
time: 18.12 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 36.20 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 36.20
Output dim: 6, lower bound: -123.9502679, upper bound: 123.9491371
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 36.20
Output dim: 6, lower bound: -123.9504628, upper bound: 123.9494092
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 36.20
Output dim: 6, lower bound: -123.9502679, upper bound: 123.9491371
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 36.20
Output dim: 6, lower bound: -123.9504628, upper bound: 123.9494092
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 36.20
Output dim: 6, lower bound: -123.9522437, upper bound: 123.9511712
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 36.20
Output dim: 6, lower bound: -123.9524486, upper bound: 123.9513490
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 36.20
Output dim: 6, lower bound: -123.9524484, upper bound: 123.9513490
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 36.20
Output dim: 6, lower bound: -123.9568805, upper bound: 123.9566513
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 36.20
Output dim: 6, lower bound: -123.9568805, upper bound: 123.9494092
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 36.20
Output dim: 6, lower bound: -123.9580803, upper bound: 123.9578957
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 36.20
Output dim: 6, lower bound: -123.9580803, upper bound: 123.9578957
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 36.20
Output dim: 6, lower bound: -123.9497413, upper bound: 123.9489758
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 36.20
Output dim: 6, lower bound: -123.9500813, upper bound: 123.9492449
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 36.20
Output dim: 6, lower bound: -123.9518894, upper bound: 123.9511223
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 36.20
Output dim: 6, lower bound: -123.9518891, upper bound: 123.9512428
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 36.20
Output dim: 6, lower bound: -123.9558915, upper bound: 123.9559217
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 36.20
Output dim: 6, lower bound: -123.9564749, upper bound: 123.9564525
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 36.20
Output dim: 6, lower bound: -123.9571480, upper bound: 123.9571984
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 36.20
Output dim: 6, lower bound: -123.9577807, upper bound: 123.9577807

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 17.11 + 603.89 = 621.00 seconds
